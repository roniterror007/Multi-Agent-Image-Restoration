import gc

import pytest
import torch


@pytest.fixture(autouse=True)
def clear_vram():
    """Automatically clears CUDA memory between tests to prevent OOM errors."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


@pytest.fixture
def sample_rgb_tensor():
    """Provides a clean [B, 3, H, W] RGB tensor on the GPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.rand(4, 3, 256, 256, device=device) * 255.0
