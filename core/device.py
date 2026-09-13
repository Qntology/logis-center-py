import torch


def detect_accelerator(device_override: str = None) -> tuple:
    if device_override:
        dev = torch.device(device_override)
        label = device_override.upper()
        if dev.type == "cuda":
            if hasattr(torch.version, "hip") and torch.version.hip is not None:
                label = f"ROCm (AMD GPU)"
            else:
                try:
                    label = f"CUDA ({torch.cuda.get_device_name(0)})"
                except Exception:
                    label = "CUDA (NVIDIA GPU)"
        return dev, label

    if torch.cuda.is_available():
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            dev = torch.device("cuda")
            return dev, "ROCm (AMD GPU)"
        dev = torch.device("cuda")
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "NVIDIA GPU"
        return dev, f"CUDA ({name})"

    return torch.device("cpu"), "CPU"


def select_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32

    try:
        major, _minor = torch.cuda.get_device_capability()
    except Exception:
        return torch.float16

    if major >= 8:
        return torch.bfloat16
    return torch.float16


def configure_backends(device: torch.device):
    if device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass


def get_vram_info() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    try:
        props = torch.cuda.get_device_properties(0)
        mem_total = props.total_memory / (1024 ** 3)
        mem_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        mem_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        mem_free = mem_total - mem_reserved
        return {
            "available": True,
            "name": props.name,
            "total_gb": round(mem_total, 2),
            "reserved_gb": round(mem_reserved, 2),
            "allocated_gb": round(mem_allocated, 2),
            "free_gb": round(mem_free, 2),
        }
    except Exception:
        return {"available": False}