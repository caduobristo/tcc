import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from src.data.spectrogram_resize import adapt_spectrogram_time


class FxDatasetAST(Dataset):
    """
    PyTorch Dataset for AST precomputed Mel spectrograms (.npy) and settings in GUITAR-FX.
    Supports in-memory RAM caching for fast epoch execution without disk I/O bottlenecks.
    """

    def __init__(
        self,
        root: str,
        excl_folders: list = None,
        processed_settings_csv: str = "proc_settings.csv",
        max_num_settings: int = 3,
        cache_in_ram: bool = False,
        transform=None,
        target_length: int = 1024, # Expected AST input_tdim
        dataset_mean: float = -4.27, # Approx values (AudioSet mean) or compute for GUITAR-FX
        dataset_std: float = 4.57,
        temporal_adaptation: str = "pad",
    ):
        self.root = os.path.abspath(root)
        self.excl_folders = excl_folders or []
        self.processed_settings_csv = processed_settings_csv
        self.max_num_settings = max_num_settings
        self.cache_in_ram = cache_in_ram
        self.transform = transform
        self.target_length = target_length
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        if temporal_adaptation not in {"pad", "bicubic"}:
            raise ValueError("temporal_adaptation must be pad or bicubic")
        self.temporal_adaptation = temporal_adaptation

        self.fx_to_label = {}
        self.label_to_fx = {}
        self.audiosamples_labels = []
        self.audiosample_to_settings = {}
        self.audiosamples_settings = []
        self.cache = {}

        self.mel_shape = ()
        self.num_fx = 0

    def init_dataset(self):
        # Map effect folders to integer class labels
        i = 0
        for folder in sorted(os.listdir(self.root)):
            folder_path = os.path.join(self.root, folder)
            if os.path.isdir(folder_path) and folder not in self.excl_folders:
                self.fx_to_label[folder] = i
                self.label_to_fx[i] = folder
                i += 1

        self.num_fx = len(self.fx_to_label)

        # Read effect parameter settings CSVs
        for folder, label_idx in self.fx_to_label.items():
            csv_path = os.path.join(self.root, folder, self.processed_settings_csv)
            if os.path.exists(csv_path):
                with open(csv_path, mode="r", encoding="utf-8") as file:
                    reader = csv.reader(file)
                    next(reader, None)
                    for row in reader:
                        if len(row) >= 2:
                            filename, fx, *settings = row
                            if filename.endswith(".wav"):
                                filename = filename[:-4]
                            settings = settings[: self.max_num_settings]
                            blanks = [-1.0] * (self.max_num_settings - len(settings))
                            settings_floats = [float(s) for s in settings] + blanks
                            self.audiosample_to_settings[filename] = settings_floats

        # Index available precomputed spectrogram files
        for folder, label_idx in self.fx_to_label.items():
            spec_dir = os.path.join(self.root, folder)
            
            if os.path.exists(spec_dir):
                for file in sorted(os.listdir(spec_dir)):
                    if not file.startswith("._") and file.endswith(".npy"):
                        filename = file[:-4]
                        exact_path = os.path.join(spec_dir, file)
                        self.audiosamples_labels.append((filename, label_idx, exact_path))
                        settings = self.audiosample_to_settings.get(
                            filename, [-1.0] * self.max_num_settings
                        )
                        self.audiosamples_settings.append((filename, settings))

        if len(self.audiosamples_labels) > 0:
            first_sample = self.__getitem__(0)
            if first_sample is not None:
                self.mel_shape = first_sample[0].shape

    def __len__(self):
        return len(self.audiosamples_labels)

    def __getitem__(self, index):
        if index < 0 or index >= len(self.audiosamples_labels):
            raise IndexError("Dataset index out of range.")

        # Return cached item directly from RAM if available
        if self.cache_in_ram and index in self.cache:
            return self.cache[index]

        filename, label, npy_path = self.audiosamples_labels[index]
        _, settings = self.audiosamples_settings[index]
        folder = self.label_to_fx[label]
        try:
            mel_array = np.load(npy_path)
            if self.transform:
                mel_tensor = self.transform(mel_array).float()
            else:
                mel_tensor = torch.tensor(mel_array, dtype=torch.float32) # (time_frames, 128)
                
            # AST Normalization (mean=0, std=0.5 approx)
            mel_tensor = (mel_tensor - self.dataset_mean) / (self.dataset_std * 2)
            
            # Default remains historical padding; the isolated ablation stretches
            # the real normalized frames, never a tensor already padded with zeros.
            mel_tensor = adapt_spectrogram_time(
                mel_tensor, self.target_length, mode=self.temporal_adaptation
            )
                
            # AST expects (time_frames, 128), and in forward pass it unsqueezes to add channel dim
        except Exception as e:
            print(f"Error loading spectrogram {npy_path}: {e}")
            return None

        label_tensor = torch.tensor(label, dtype=torch.long)
        settings_tensor = torch.tensor(settings, dtype=torch.float32)

        item = (mel_tensor, label_tensor, settings_tensor, filename, index)

        if self.cache_in_ram:
            self.cache[index] = item

        return item
