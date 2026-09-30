import argparse
import csv
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# Add src to import path
project_root = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(project_root / 'src'))

import dataset.dataset as dataset_mod
import datasplit.datasplit as datasplit
import model.models as models
import trainer.trainer as trainer


def append_summary_csv(path: Path, row: dict):
    header = [
        'timestamp',
        'run_name',
        'dataset_root',
        'device',
        'epochs_requested',
        'epochs_completed',
        'batch_size',
        'lr',
        'train_size',
        'val_size',
        'test_size',
        'best_epoch',
        'best_train_acc',
        'best_val_acc',
        'best_test_acc',
        'last_epoch',
        'last_train_acc',
        'last_val_acc',
        'last_test_acc',
        'total_seconds',
        'best_model',
        'last_model',
        'history_csv',
    ]
    write_header = not path.exists()
    with path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_summary_md(path: Path, row: dict):
    with path.open('a', encoding='utf-8') as f:
        if path.stat().st_size == 0:
            f.write('# Training Runs Summary\n\n')
        f.write(f"## {row['timestamp']} - {row['run_name']}\n")
        f.write(f"- dataset_root: `{row['dataset_root']}`\n")
        f.write(f"- device: `{row['device']}`\n")
        f.write(
            f"- epochs: requested={row['epochs_requested']}, completed={row['epochs_completed']}, "
            f"batch_size={row['batch_size']}, lr={row['lr']}\n"
        )
        f.write(
            f"- split sizes: train={row['train_size']}, val={row['val_size']}, "
            f"test={row['test_size']}\n"
        )
        f.write(
            f"- best epoch: {row['best_epoch']} | train={row['best_train_acc']}% "
            f"val={row['best_val_acc']}% test={row['best_test_acc']}%\n"
        )
        f.write(
            f"- last epoch: {row['last_epoch']} | train={row['last_train_acc']}% "
            f"val={row['last_val_acc']}% test={row['last_test_acc']}%\n"
        )
        f.write(f"- total_seconds: {row['total_seconds']}\n")
        f.write(f"- best_model: `{row['best_model']}`\n")
        f.write(f"- last_model: `{row['last_model']}`\n")
        f.write(f"- history_csv: `{row['history_csv']}`\n\n")


def append_summary_txt(path: Path, row: dict):
    with path.open('a', encoding='utf-8') as f:
        if path.stat().st_size == 0:
            f.write(
                'timestamp | run_name | dataset_root | device | epochs_requested | epochs_completed | '
                'batch_size | lr | train_size | val_size | test_size | '
                'best_epoch | best_train_acc | best_val_acc | best_test_acc | '
                'last_epoch | last_train_acc | last_val_acc | last_test_acc | '
                'total_seconds | best_model | last_model | history_csv\n'
            )
        f.write(
            f"{row['timestamp']} | {row['run_name']} | {row['dataset_root']} | {row['device']} | "
            f"{row['epochs_requested']} | {row['epochs_completed']} | {row['batch_size']} | {row['lr']} | "
            f"{row['train_size']} | {row['val_size']} | {row['test_size']} | "
            f"{row['best_epoch']} | {row['best_train_acc']} | {row['best_val_acc']} | {row['best_test_acc']} | "
            f"{row['last_epoch']} | {row['last_train_acc']} | {row['last_val_acc']} | {row['last_test_acc']} | "
            f"{row['total_seconds']} | {row['best_model']} | {row['last_model']} | {row['history_csv']}\n"
        )


def main():
    parser = argparse.ArgumentParser(description='Train FxNet on a full dataset root (effect folders with mel features).')
    parser.add_argument('--run-name', type=str, default='fxnet_mono_cont_full_gpu')
    parser.add_argument(
        '--dataset-root',
        type=str,
        default='..\\data\\GUITAR-FX\\Mono_Continuous\\Features',
        help='Absolute path or path relative to project root (gfx-classifier).',
    )
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exclude', nargs='*', default=['MT2'])
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_root_arg = Path(args.dataset_root)
    root = dataset_root_arg if dataset_root_arg.is_absolute() else (project_root / dataset_root_arg)
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f'Dataset root not found: {root}')

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    fx_dataset = dataset_mod.FxDataset(
        root=str(root),
        excl_folders=args.exclude,
        spectra_folder='mel_22050_1024_512',
        processed_settings_csv='proc_settings.csv',
        max_num_settings=6,
        transform=transform,
    )
    fx_dataset.init_dataset()

    split = datasplit.DataSplit(fx_dataset, test_train_split=0.8, val_train_split=0.1, shuffle=True)
    train_loader, val_loader, test_loader = split.get_split(batch_size=args.batch_size)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.FxNet(n_classes=fx_dataset.num_fx).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print('=== Full Training Config ===')
    print(f'device: {device}')
    if torch.cuda.is_available():
        print(f'gpu: {torch.cuda.get_device_name(0)}')
    print(f'dataset root: {root}')
    print(f'dataset size: {len(fx_dataset)}')
    print(f'train/val/test: {len(split.train_sampler)}/{len(split.val_sampler)}/{len(split.test_sampler)}')
    print(f'mel shape: {fx_dataset.mel_shape}')
    print(f'num classes: {fx_dataset.num_fx}')
    print(f'epochs: {args.epochs}, batch_size: {args.batch_size}, lr: {args.lr}')

    best_val_acc = -1.0
    best_state = None
    history = []
    run_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_loss, train_correct, _ = trainer.train_fx_net(model, optimizer, train_loader, split.train_sampler, epoch=epoch, loss_function=loss_fn, device=device)
        val_loss, val_correct, _ = trainer.val_fx_net(model, val_loader, split.val_sampler, loss_function=loss_fn, device=device)
        test_loss, test_correct, _ = trainer.test_fx_net(model, test_loader, split.test_sampler, loss_function=loss_fn, device=device)

        train_acc = 100.0 * train_correct / len(split.train_sampler)
        val_acc = 100.0 * val_correct / len(split.val_sampler)
        test_acc = 100.0 * test_correct / len(split.test_sampler)
        elapsed = time.time() - epoch_start

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'epoch_seconds': elapsed,
        })

        print(
            f"EPOCH {epoch} SUMMARY | "
            f"train_acc={train_acc:.2f}% val_acc={val_acc:.2f}% test_acc={test_acc:.2f}% "
            f"elapsed={elapsed:.2f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    total_seconds = time.time() - run_start

    models_dir = project_root.parent / 'models_and_results' / 'models'
    results_dir = project_root.parent / 'models_and_results' / 'results'
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    stamp = int(time.time())
    model_path = models_dir / f'{args.run_name}_{stamp}.pt'
    last_model_path = models_dir / f'{args.run_name}_last_{stamp}.pt'
    history_path = results_dir / f'{args.run_name}_history_{stamp}.csv'
    summary_csv_path = results_dir / 'training_runs_summary.csv'
    summary_md_path = results_dir / 'training_runs_summary.md'
    summary_txt_path = results_dir / 'training_runs_summary.txt'

    if best_state is not None:
        torch.save(best_state, model_path)
    torch.save(model.state_dict(), last_model_path)

    with history_path.open('w', encoding='utf-8') as f:
        f.write('epoch,train_loss,val_loss,test_loss,train_acc,val_acc,test_acc,epoch_seconds\n')
        for row in history:
            f.write(
                f"{row['epoch']},{row['train_loss']:.6f},{row['val_loss']:.6f},{row['test_loss']:.6f},"
                f"{row['train_acc']:.4f},{row['val_acc']:.4f},{row['test_acc']:.4f},{row['epoch_seconds']:.4f}\n"
            )

    best_row = max(history, key=lambda r: r['val_acc']) if history else None
    last_row = history[-1] if history else None
    summary_row = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'run_name': args.run_name,
        'dataset_root': str(root),
        'device': str(device),
        'epochs_requested': args.epochs,
        'epochs_completed': len(history),
        'batch_size': args.batch_size,
        'lr': args.lr,
        'train_size': len(split.train_sampler),
        'val_size': len(split.val_sampler),
        'test_size': len(split.test_sampler),
        'best_epoch': best_row['epoch'] if best_row else '',
        'best_train_acc': f"{best_row['train_acc']:.4f}" if best_row else '',
        'best_val_acc': f"{best_row['val_acc']:.4f}" if best_row else '',
        'best_test_acc': f"{best_row['test_acc']:.4f}" if best_row else '',
        'last_epoch': last_row['epoch'] if last_row else '',
        'last_train_acc': f"{last_row['train_acc']:.4f}" if last_row else '',
        'last_val_acc': f"{last_row['val_acc']:.4f}" if last_row else '',
        'last_test_acc': f"{last_row['test_acc']:.4f}" if last_row else '',
        'total_seconds': f"{total_seconds:.2f}",
        'best_model': str(model_path),
        'last_model': str(last_model_path),
        'history_csv': str(history_path),
    }
    append_summary_csv(summary_csv_path, summary_row)
    append_summary_md(summary_md_path, summary_row)
    append_summary_txt(summary_txt_path, summary_row)

    print('=== Full Training Complete ===')
    print(f'total_seconds: {total_seconds:.2f}')
    print(f'best_val_acc: {best_val_acc:.2f}')
    print(f'best_model: {model_path}')
    print(f'last_model: {last_model_path}')
    print(f'history_csv: {history_path}')
    print(f'summary_csv: {summary_csv_path}')
    print(f'summary_md: {summary_md_path}')
    print(f'summary_txt: {summary_txt_path}')


if __name__ == '__main__':
    main()
