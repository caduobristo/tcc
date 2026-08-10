import os
import time
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataset import FxDataset
from src.data.datasplit import DataSplit
from src.models.fxnet import FxNet
from src.utils.metrics import compute_metrics, plot_confusion_matrix
from src.utils.logger import setup_logger
from src.utils.device import get_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trains baseline FxNet model on GUITAR-FX dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/baseline_mono_disc.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Dataset root path (overrides config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda', 'dml', or 'cpu')",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Batch limit per epoch for quick test",
    )
    return parser.parse_args()


class DirectMLAdam(optim.Optimizer):
    """Adam optimizer using basic ops natively supported on DirectML GPU."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                bias_correction1 = 1.0 - (beta1 ** state["step"])
                bias_correction2 = 1.0 - (beta2 ** state["step"])

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                step_size = lr / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss


def main():
    args = parse_args()

    config_path = os.path.abspath(args.config)
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    output_cfg = cfg.get("output", {})

    dataset_root = args.data_dir or data_cfg.get(
        "dataset_root", "../datasets/GUITAR-FX-DIST/Mono_Discrete"
    )
    spectra_folder = data_cfg.get("spectra_folder", "mel_22050_1024_512")
    excl_folders = data_cfg.get("excl_folders", ["TS9", "MT2"])
    seed = data_cfg.get("seed", 42)

    epochs = args.epochs or train_cfg.get("epochs", 30)
    batch_size = args.batch_size or train_cfg.get("batch_size", 100)
    lr = args.lr or train_cfg.get("learning_rate", 0.001)

    checkpoints_dir = output_cfg.get("checkpoints_dir", "checkpoints/baseline")
    results_dir = output_cfg.get("results_dir", "results/baseline")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    logger = setup_logger("train_baseline", os.path.join(results_dir, "train.log"))

    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    cache_in_ram = data_cfg.get("cache_in_ram", True)
    logger.info(f"Loading dataset from: {os.path.abspath(dataset_root)} (RAM caching: {cache_in_ram})")
    dataset = FxDataset(
        root=dataset_root,
        excl_folders=excl_folders,
        spectra_folder=spectra_folder,
        cache_in_ram=cache_in_ram,
    )
    dataset.init_dataset()

    logger.info(f"Total dataset samples: {len(dataset)}")
    logger.info(f"Number of effect classes: {dataset.num_fx}")

    split = DataSplit(
        dataset,
        test_train_split=data_cfg.get("test_train_split", 0.8),
        val_train_split=data_cfg.get("val_train_split", 0.1),
        shuffle=data_cfg.get("shuffle", True),
        seed=seed,
    )
    train_loader, val_loader, test_loader = split.get_split(
        batch_size=batch_size, num_workers=train_cfg.get("num_workers", 0)
    )

    logger.info(
        f"Sample split -> Train: {len(split.train_sampler)} | Val: {len(split.val_indices)} | Test: {len(split.test_indices)}"
    )

    model = FxNet(n_classes=dataset.num_fx).to(device)

    if "privateuseone" in str(device) or "dml" in str(device):
        optimizer = DirectMLAdam(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"FxNet initialized with {trainable_params:,} trainable parameters.")

    best_val_acc = 0.0
    epoch_times = []

    total_start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, data in enumerate(train_loader):
            if data is None:
                continue
            inputs, labels, _, _, _ = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct_train += preds.eq(labels).sum().item()
            total_train += labels.size(0)

            if args.max_batches and (batch_idx + 1) >= args.max_batches:
                break

        train_loss = running_loss / max(total_train, 1)
        train_acc = (correct_train / max(total_train, 1)) * 100

        model.eval()
        val_loss_total = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                if data is None:
                    continue
                inputs, labels, _, _, _ = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss_total += loss.item() * inputs.size(0)

                preds = outputs.argmax(dim=1)
                correct_val += preds.eq(labels).sum().item()
                total_val += labels.size(0)

                if args.max_batches and (batch_idx + 1) >= args.max_batches:
                    break

        val_loss = val_loss_total / max(total_val, 1)
        val_acc = (correct_val / max(total_val, 1)) * 100

        epoch_duration = time.time() - epoch_start
        epoch_times.append(epoch_duration)

        logger.info(
            f"Epoch [{epoch+1:02d}/{epochs:02d}] ({epoch_duration:.2f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_checkpoint_path = os.path.join(checkpoints_dir, "best_model.pt")
            torch.save(model.state_dict(), best_checkpoint_path)
            logger.info(f"  -> Best model saved (Val Acc: {val_acc:.2f}%)")

    total_duration = time.time() - total_start_time
    avg_epoch_time = sum(epoch_times) / max(len(epoch_times), 1)

    logger.info("=" * 60)
    logger.info(f"TRAINING COMPLETED in {total_duration/60:.2f} minutes.")
    logger.info(f"Average time per epoch: {avg_epoch_time:.2f} seconds.")
    logger.info(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    logger.info("=" * 60)

    best_checkpoint_path = os.path.join(checkpoints_dir, "best_model.pt")
    if os.path.exists(best_checkpoint_path):
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))

    model.eval()
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            if data is None:
                continue
            inputs, labels, _, _, _ = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

            if args.max_batches and (batch_idx + 1) >= args.max_batches:
                break

    class_names = [dataset.label_to_fx[i] for i in range(dataset.num_fx)]
    test_metrics = compute_metrics(test_targets, test_preds, class_names=class_names)

    logger.info("FINAL TEST RESULTS:")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    logger.info(f"Test F1-Score: {test_metrics['macro_f1']*100:.2f}%")

    cm_path = os.path.join(results_dir, "confusion_matrix_train.png")
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        classes=class_names,
        save_path=cm_path,
        normalize=True,
        title="Confusion Matrix - FxNet Training",
    )
    logger.info(f"Training confusion matrix saved to: {os.path.abspath(cm_path)}")


if __name__ == "__main__":
    main()
