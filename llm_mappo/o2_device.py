"""Shared device provenance checks for O2 owner entry points."""

import torch


def device_provenance(logical_device: str) -> dict:
    """Record the actual device used by a diagnostic or formal O2 artifact."""
    if logical_device == "cpu":
        return {
            "logical_device": "cpu",
            "cuda_available": False,
            "device_name": None,
            "torch": torch.__version__,
        }
    device = torch.device(logical_device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("O2 CUDA provenance requires an available CUDA device.")
    index = 0 if device.index is None else device.index
    return {
        "logical_device": str(device),
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(index),
        "torch": torch.__version__,
    }
