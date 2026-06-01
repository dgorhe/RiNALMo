from contextlib import contextmanager

import torch


def select_inference_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@contextmanager
def inference_autocast(device: torch.device):
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    elif device.type == "mps":
        with torch.autocast(device_type="mps", dtype=torch.float16):
            yield
    else:
        yield
