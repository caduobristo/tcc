"""Run the documented head-only protocol; requires an explicit --run flag."""
from pathlib import Path
from datetime import datetime
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('MPLBACKEND','Agg')
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
sys.stderr.reconfigure(encoding='utf-8',errors='replace')


class Tee:
    def __init__(self, stream, log):
        self.stream,self.log = stream,log

    def write(self,value):
        self.log.write(value)
        return self.stream.write(value)

    def flush(self):
        self.log.flush()
        self.stream.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--plan',type=Path,default=ROOT/'configs/linear_probe/consolidation_mono_disc.json')
    args = parser.parse_args()
    if not args.run:
        print(args.plan.read_text(encoding='utf-8'))
        print('Training disabled; --run is required.')
    else:
        import torch
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        from src.training.consolidation import run_consolidation
        log_path = ROOT/'data/audits/transfer_learning'/f'consolidation_{datetime.now():%Y%m%d_%H%M%S}.log'
        log_path.parent.mkdir(parents=True,exist_ok=True)
        with log_path.open('w',encoding='utf-8',buffering=1) as log:
            original_out,original_err = sys.stdout,sys.stderr
            sys.stdout,sys.stderr = Tee(original_out,log),Tee(original_err,log)
            try:
                run_consolidation(args.plan,allow_training=True)
            finally:
                sys.stdout,sys.stderr = original_out,original_err
