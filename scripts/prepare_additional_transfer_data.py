"""Restore selected official WAVs using verified, disposable download volumes.

Existing user archives are never removed. Only this script's scratch downloads
are discarded, after extraction and a full CRC check against the archive index.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_transfer_data import RECORDS, download, file_hash
from src.data.paths import SCENARIOS, data_root
from src.data.transfer import EFFECTS


class RequestPacer:
    """Shared spacing and server cooldown across all volume download threads."""
    def __init__(self, interval=1.25):
        self.interval = interval
        self.next_allowed = 0.
        self.lock = threading.Lock()

    def wait(self):
        while True:
            with self.lock:
                now = time.monotonic()
                delay = self.next_allowed-now
                if delay <= 0:
                    self.next_allowed = now+self.interval
                    return
            time.sleep(min(delay,5))

    def defer(self, seconds):
        with self.lock:
            self.next_allowed = max(self.next_allowed,time.monotonic()+seconds)


def retry_after_seconds(headers, attempt):
    value = headers.get('Retry-After')
    if value:
        try:
            return max(1.,float(value))
        except ValueError:
            try:
                return max(1.,(parsedate_to_datetime(value)-datetime.now(timezone.utc)).total_seconds())
            except (ValueError,TypeError):
                pass
    return min(60*2**attempt,600)


RANGE_PACER = RequestPacer()


def remaining_download_bytes(files, folder, block_bytes=4*2**20):
    """Count missing ranges rather than charging for complete resumed volumes."""
    missing = 0
    for file in files:
        name, size = file['key'], file['size']
        if Path(name).name != name:
            raise ValueError('Unsafe source filename')
        destination = folder/name
        if destination.exists():
            if destination.stat().st_size != size:
                raise ValueError('Existing scratch volume has unexpected size')
            continue  # download_ranges still verifies its published checksum
        prefix = folder/(name+'.partial')
        prefix_size = prefix.stat().st_size if prefix.exists() else 0
        if prefix_size > size:
            raise ValueError('Oversized partial download')
        if size < block_bytes:
            missing += size-prefix_size
            continue
        for index, start in enumerate(range(0, size, block_bytes)):
            length = min(block_bytes, size-start)
            block = folder/(name+'.blocks')/f'{index:05d}.part'
            if (block.exists() and block.stat().st_size == length) or start+length <= prefix_size:
                continue
            missing += length
    return missing


def download_ranges(file, folder, connections=4, block_bytes=4*2**20, max_blocks_per_request=4):
    """Bounded parallel ranges, resumable blocks, then published full-volume MD5."""
    name = file['key']
    if Path(name).name != name:
        raise ValueError('Unsafe source filename')
    if file['size'] < block_bytes:
        return download(file,folder)
    dest = folder/name
    algorithm, expected = file['checksum'].split(':',1)
    if dest.exists():
        if dest.stat().st_size != file['size'] or file_hash(dest,algorithm)!=expected:
            raise ValueError('Existing scratch volume failed checksum')
        print('Verified existing',name,flush=True)
        return
    chunks = folder/(name+'.blocks')
    chunks.mkdir(exist_ok=True)
    prefix = folder/(name+'.partial')
    prefix_size = prefix.stat().st_size if prefix.exists() else 0
    count = (file['size']+block_bytes-1)//block_bytes

    pending = []
    for index in range(count):
        start = index*block_bytes
        end = min(start+block_bytes,file['size'])-1
        block = chunks/f'{index:05d}.part'
        if block.exists() and block.stat().st_size==end-start+1:
            continue
        temporary = block.with_suffix('.downloading')
        if end < prefix_size:
            with prefix.open('rb') as stream:
                stream.seek(start)
                temporary.write_bytes(stream.read(end-start+1))
            temporary.replace(block)
            continue
        pending.append(index)
    # Keep the on-disk 4 MiB resume format, but request consecutive missing
    # blocks together (at most 16 MiB) to avoid hitting server request limits.
    groups = []
    for index in pending:
        if groups and len(groups[-1]) < max_blocks_per_request and index == groups[-1][-1]+1:
            groups[-1].append(index)
        else:
            groups.append([index])

    def fetch(indices):
        start = indices[0]*block_bytes
        end = min((indices[-1]+1)*block_bytes,file['size'])-1
        temporary = chunks/f'{indices[0]:05d}.downloading'
        for attempt in range(20):
            try:
                RANGE_PACER.wait()
                url = file['links']['self']+f'?tcc_range={start}-{end}&attempt={attempt}'
                request = urllib.request.Request(url,headers={
                    'Range':f'bytes={start}-{end}','User-Agent':'TCC-dataset-preparation/1.0'})
                with urllib.request.urlopen(request,timeout=60) as response:
                    if response.status!=206 or response.headers.get('Content-Range')!=f"bytes {start}-{end}/{file['size']}":
                        raise OSError(f'Source did not honor exact range: {response.status}, {response.headers.get("Content-Range")}')
                    with temporary.open('wb') as stream:
                        shutil.copyfileobj(response,stream,length=256*1024)
                if temporary.stat().st_size!=end-start+1:
                    raise OSError('Incomplete block')
                with temporary.open('rb') as stream:
                    for index in indices:
                        block = chunks/f'{index:05d}.part'
                        payload = stream.read(min(block_bytes,file['size']-index*block_bytes))
                        piece = block.with_suffix('.piece')
                        piece.write_bytes(payload)
                        piece.replace(block)
                temporary.unlink()
                return
            except Exception as exc:
                if isinstance(exc,urllib.error.HTTPError) and exc.code==429:
                    delay = retry_after_seconds(exc.headers,attempt)
                    RANGE_PACER.defer(delay)
                    print(f'RATE_LIMIT {name}: shared cooldown {delay:.1f}s, honoring Retry-After',flush=True)
                print(f'RETRY {name} blocks={indices[0]}-{indices[-1]} attempt={attempt+1}: {type(exc).__name__}: {exc}',flush=True)
                if attempt==19:
                    raise
                time.sleep(min(2**attempt,10))
    last_report = time.monotonic()
    with ThreadPoolExecutor(max_workers=connections) as pool:
        futures = {pool.submit(fetch,indices):indices for indices in groups}
        completed = count-len(pending)
        for future in as_completed(futures):
            future.result()
            completed += len(futures[future])
            if time.monotonic()-last_report>30:
                print(f'{name}: {completed}/{count} verified-length blocks',flush=True)
                last_report=time.monotonic()
    assembled = folder/(name+'.assembling')
    digest = __import__('hashlib').new(algorithm)
    with assembled.open('wb') as output:
        for index in range(count):
            payload = (chunks/f'{index:05d}.part').read_bytes()
            digest.update(payload)
            output.write(payload)
    if assembled.stat().st_size!=file['size'] or digest.hexdigest()!=expected:
        raise ValueError(f'Published checksum mismatch: {name}')
    assembled.replace(dest)
    # These are precisely enumerated temporary files owned by this download.
    if chunks.resolve().parent!=folder.resolve():
        raise ValueError('Unsafe block cleanup')
    for index in range(count):
        (chunks/f'{index:05d}.part').unlink()
        for suffix in ('.downloading','.piece'):
            temporary = chunks/f'{index:05d}{suffix}'
            if temporary.exists():
                temporary.unlink()
    chunks.rmdir()
    if prefix.exists():
        prefix.unlink()
    print('Downloaded and MD5 verified',name,flush=True)


def selected_members(listing, scenario):
    members = []
    for block in listing.split('----------\n', 1)[1].split('\n\n'):
        fields = dict(line.split(' = ', 1) for line in block.splitlines() if ' = ' in line)
        if fields.get('Folder') != '-':
            continue
        path = Path(fields['Path'].replace('\\', '/'))
        if path.parts[0] != scenario or path.is_absolute() or '..' in path.parts:
            raise ValueError(f'Unsafe archive member: {path}')
        wav = len(path.parts) >= 4 and path.parts[1] == 'Audio' and path.parts[2] in EFFECTS and path.suffix == '.wav'
        settings = path.name == 'proc_settings.csv'
        if wav or settings:
            members.append({'path':path.as_posix(), 'size':int(fields['Size']), 'crc32':fields['CRC'].lower()})
    if {Path(m['path']).parts[2] for m in members if m['path'].endswith('.wav')} != set(EFFECTS):
        raise ValueError('Official archive lacks a selected class')
    return members


def prepare(scenario, workers=3, discard_downloads=False):
    """One writer per scenario, including concurrently prefetched datasets."""
    lock = data_root()/'audits/transfer_learning'/f'prepare_{scenario}.lock'
    lock.parent.mkdir(parents=True,exist_ok=True)
    while True:
        try:
            with lock.open('x',encoding='utf-8') as stream:
                stream.write(str(os.getpid()))
            break
        except FileExistsError:
            import psutil
            pid = int(lock.read_text(encoding='utf-8'))
            if not psutil.pid_exists(pid):
                raise RuntimeError(f'Stale preparation lock: {lock}; inspect the preserved scratch data')
            time.sleep(5)
            receipt = data_root()/'archives/guitar_fx_dist/official'/SCENARIOS[scenario]/'selected_wavs_receipt.json'
            if receipt.exists() and not lock.exists():
                return json.loads(receipt.read_text(encoding='utf-8'))
    try:
        receipt = data_root()/'archives/guitar_fx_dist/official'/SCENARIOS[scenario]/'selected_wavs_receipt.json'
        if receipt.exists():
            return json.loads(receipt.read_text(encoding='utf-8'))
        return _prepare(scenario,workers,discard_downloads)
    finally:
        if lock.read_text(encoding='utf-8')!=str(os.getpid()):
            raise RuntimeError('Preparation lock ownership changed')
        lock.unlink()


def _prepare(scenario, workers=3, discard_downloads=False):
    started = time.perf_counter()
    root = data_root()
    name, record = SCENARIOS[scenario], RECORDS[scenario]
    official = root/'archives/guitar_fx_dist/official'/name
    official.mkdir(parents=True, exist_ok=True)
    receipt = official/'selected_wavs_receipt.json'
    if receipt.exists():
        raise FileExistsError('Preparation already completed; inspect the receipt and reuse the audited manifest')
    source_path = official/'source_record.json'
    if source_path.exists():
        source = json.loads(source_path.read_text(encoding='utf-8'))
    else:
        with urllib.request.urlopen(f'https://zenodo.org/api/records/{record}', timeout=60) as response:
            source = json.load(response)
        source_path.write_text(json.dumps(source, indent=2), encoding='utf-8')
    if str(source['id']) != record:
        raise ValueError('Record identity mismatch')
    files = sorted([f for f in source['files'] if f['key'].startswith(name+'.') or f['key']=='README.md'], key=lambda f:f['key'])
    # This directory is owned by this new workflow, not the user's archive folder.
    scratch = (root/'audits/transfer_learning/source_downloads'/name).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    missing_bytes = remaining_download_bytes(files, scratch)
    reserve = 12*2**30  # subsequent embeddings, metadata and disk reserve
    # Download and extraction have separate peaks. Reassembly briefly keeps
    # complete blocks alongside each new volume; extraction is checked below
    # against the archive's exact selected members, including resumed files.
    assembly_peak = sum(sorted((f['size'] for f in files if not (scratch/f['key']).exists()), reverse=True)[:workers])
    if shutil.disk_usage(root).free < missing_bytes + assembly_peak + reserve:
        raise OSError('Insufficient space for missing ranges + volume reassembly + 12 GiB reserve')
    print(f'DATA_PREPARATION {scenario}: {sum(f["size"] for f in files)/1e9:.2f} GB official downloads', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(lambda f:download_ranges(f,scratch), files))
    binary = shutil.which('7z') or 'C:/Program Files/7-Zip/7z.exe'
    archive = scratch/(name+'.zip')
    listing = subprocess.run([binary,'l','-slt',str(archive)],check=True,capture_output=True,text=True,encoding='utf-8').stdout.replace('\r\n','\n')
    members = selected_members(listing,name)
    output = (root/'raw/guitar_fx_dist').resolve()
    for member in members:
        if not (output/member['path']).resolve().is_relative_to(output):
            raise ValueError('Unsafe extraction path')
    required = sum(m['size'] for m in members if not (output/m['path']).exists())
    if shutil.disk_usage(root).free < required + reserve:
        raise OSError('Insufficient extraction space')
    includes = scratch/'selected_members.txt'
    includes.write_text('\n'.join(m['path'] for m in members)+'\n',encoding='utf-8')
    print(f'EXTRACTING {len(members)} WAV/CSV members, {required/1e9:.2f} GB',flush=True)
    output.mkdir(parents=True,exist_ok=True)
    subprocess.run([binary,'x',str(archive),f'-o{output}','-aos','-bsp0','-scsUTF-8',f'-i@{includes}'],check=True)

    def verify(member):
        path = output/member['path']
        payload = path.read_bytes()
        if len(payload)!=member['size'] or f'{zlib.crc32(payload):08x}'!=member['crc32']:
            raise ValueError(f'Extracted member failed source CRC: {path}')
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(verify,members))
    (official/'selected_members.json').write_text(json.dumps(members,indent=2),encoding='utf-8')
    result = {'scenario':scenario,'source':f'https://zenodo.org/records/{record}',
              'completed_at_utc':datetime.now(timezone.utc).isoformat(),
              'elapsed_seconds':time.perf_counter()-started,'archive_checksums_verified':True,
              'volumes':[{'name':f['key'],'bytes':f['size'],'checksum':f['checksum']} for f in files],
              'extracted_members':len(members),'selected_wavs':sum(m['path'].endswith('.wav') for m in members),
              'extracted_bytes':sum(m['size'] for m in members),'all_extracted_crc_verified':True,
              'raw_root':str(output/name),'download_volumes_retained':not discard_downloads,
              'scope':'13 effect classes; MT2/NoFX and baseline NPYs not extracted'}
    (official/'README.md').write_bytes((scratch/'README.md').read_bytes())
    receipt.write_text(json.dumps(result,indent=2),encoding='utf-8')
    if discard_downloads:
        allowed = (root/'audits/transfer_learning/source_downloads').resolve()
        if not scratch.is_relative_to(allowed):
            raise ValueError('Scratch cleanup escaped workflow directory')
        for file in files:
            target = (scratch/file['key']).resolve()
            if target.parent != scratch:
                raise ValueError('Unsafe scratch cleanup target')
            target.unlink()  # precisely enumerated workflow-owned downloads only
        includes.unlink()
    print('DATA_PREPARATION_COMPLETE',json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario',choices=['mono_cont','poly_cont','poly_disc'],required=True)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--discard-downloads',action='store_true')
    args = parser.parse_args()
    prepare(args.scenario,args.workers,args.discard_downloads)
