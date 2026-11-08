import torch
import psutil

def get_batch_settings():
    """
    Automatically sets BATCH_SIZE and ACCUMULATION_STEPS 
    based on available hardware VRAM or system RAM.
    """
    if torch.cuda.is_available():
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
        device_name = torch.cuda.get_device_name(0)
        print(f"Training on GPU: {device_name} ({vram_gb:.1f} GB VRAM)")
        
        if vram_gb < 10:
            batch_size = 64
            accumulation_steps = 16  # effective_batch size = 4
        elif vram_gb < 18:
            batch_size = 144
            accumulation_steps = 12  # effective_batch size = 12
        elif vram_gb < 30:

            batch_size = 256
            accumulation_steps = 8   # effective_batch size = 32
        else:
            batch_size = 384
            accumulation_steps = 8   # effective_batch size = 48
            
    else:
        sys_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        print(f"Training on CPU. System RAM: {sys_ram_gb:.1f} GB.")
        if sys_ram_gb<32:        
            batch_size = 16
            accumulation_steps = 8
        else:
            batch_size=64
            accumulation_steps=16

    return batch_size, accumulation_steps

BATCH_SIZE, ACCUMULATION_STEPS = auto_detect_batch_settings()

print(f"   • Total Effective Batch Size : {BATCH_SIZE}")
print(f"   • Accumulation Steps         : {ACCUMULATION_STEPS}")