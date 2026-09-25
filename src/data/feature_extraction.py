import os
import glob
import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm
from joblib import Parallel, delayed

def get_fbank(waveform, sample_rate, target_sr=16000):
    """
    Converte o áudio bruto em Log-Mel Spectrogram usando a implementação Kaldi.
    Essa é exatamente a mesma representação interna esperada pelos feature_extractors 
    do Hugging Face (AST, AudioMAE) e pelo pacote do PaSST.
    """
    # Converter para mono caso tenha mais de um canal
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample para a taxa alvo (16k ou 32k)
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
    
    # A implementação Fbank do Kaldi espera que o áudio (em float de -1 a 1)
    # seja escalado para o range de 16-bits.
    waveform = waveform * 32768.0
    
    # Computar o Fbank (Log-Mel Spectrogram)
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=target_sr,
        use_energy=False,
        window_type='hanning',
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10 # Padrão dos Transformers: janela a cada 10ms
    )
    return fbank

def process_single_audio(wav_path, output_path, target_sr):
    """
    Lê um áudio, processa e salva como arquivo .npy.
    """
    # Ignora se já foi processado anteriormente (ajuda a pausar e retomar)
    if os.path.exists(output_path):
        return None
        
    try:
        # Usa backend alternativo caso o torchcodec não esteja presente
        try:
            waveform, sr = torchaudio.load(wav_path, backend="soundfile")
        except Exception:
            waveform, sr = torchaudio.load(wav_path)
            
        fbank = get_fbank(waveform, sr, target_sr)
        
        # Salva o tensor em formato numpy
        # O shape final do array será (time_frames, 128 mels)
        np.save(output_path, fbank.numpy())
        return None
    except Exception as e:
        return f"Erro em {os.path.basename(wav_path)}: {str(e)}"

def process_dataset_parallel(dataset_root, target_sr=16000, output_folder_name="mel_16", excl_folders=None, num_workers=8):
    """
    Varre a estrutura do dataset e roda a conversão usando múltiplos núcleos da CPU.
    """
    if excl_folders is None:
        excl_folders = []
        
    audio_base_path = os.path.join(dataset_root, "Audio")
    
    if not os.path.exists(audio_base_path):
        print(f"Erro: A pasta 'Audio' não foi encontrada em {dataset_root}")
        return
        
    # Encontra todas as pastas dos efeitos (classes) em 'Audio'
    # Agora inclui todos (incluindo MT2 e _NoFX) por padrão
    effect_folders = [f for f in os.listdir(audio_base_path) 
                     if os.path.isdir(os.path.join(audio_base_path, f)) 
                     and f not in excl_folders]
    
    # Define a base de saída: ex: ../dataset/GUITAR-FX-DIST/mel_16/Mono_Discrete
    dataset_root_clean = dataset_root.rstrip(os.sep)
    subset_name = os.path.basename(dataset_root_clean)
    global_dataset_root = os.path.dirname(dataset_root_clean)
    output_base_path = os.path.join(global_dataset_root, output_folder_name, subset_name)
    
    tasks = []
    
    for folder in effect_folders:
        audio_folder_path = os.path.join(audio_base_path, folder)
        
        # O destino final dos arrays deve ficar na pasta do efeito dentro de mel_16/Subset/
        output_dir = os.path.join(output_base_path, folder)
        os.makedirs(output_dir, exist_ok=True)
        
        # Procura por todos os arquivos .wav na pasta de efeito atual
        wav_files = glob.glob(os.path.join(audio_folder_path, "**", "*.wav"), recursive=True)
            
        for wav_path in wav_files:
            # Obtém o nome do arquivo, ex: 'audio001.wav' -> 'audio001.npy'
            filename = os.path.basename(wav_path).replace(".wav", ".npy")
            output_path = os.path.join(output_dir, filename)
            tasks.append((wav_path, output_path, target_sr))
            
    print(f"Encontrados {len(tasks)} arquivos de áudio para processar na taxa {target_sr}Hz.")
    
    # Roda em paralelo e coleta os resultados
    results = Parallel(n_jobs=num_workers)(
        delayed(process_single_audio)(wav_path, out_path, sr) 
        for wav_path, out_path, sr in tqdm(tasks, desc=f"Gerando {output_folder_name}")
    )
    
    errors = [r for r in results if r is not None]
    if errors:
        print(f"Processamento concluído com {len(errors)} erros.")
        print(f"Primeiro erro: {errors[0]}")
        print("Dica: Se o erro for 'TorchCodec is required...', rode 'pip install soundfile torchcodec' no terminal.")
    else:
        print("Processamento concluído com sucesso e sem erros!")
