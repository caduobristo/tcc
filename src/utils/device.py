import torch


def get_device(requested_device: str = None):
    """
    Returns PyTorch compute device (CUDA, DirectML for AMD GPUs on Windows, or CPU).
    """
    if requested_device:
        req_lower = requested_device.lower()
        if req_lower in ["dml", "directml"]:
            try:
                import torch_directml
                return torch_directml.device()
            except ImportError:
                print("[WARNING] torch-directml is not installed.")
                print("          To use AMD Radeon GPUs on Windows, run: pip install torch-directml")
                return torch.device("cpu")
        return torch.device(requested_device)

    # Auto-detection: 1) CUDA -> 2) DirectML (AMD GPU) -> 3) CPU
    if torch.cuda.is_available():
        return torch.device("cuda")

    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        pass

    return torch.device("cpu")
