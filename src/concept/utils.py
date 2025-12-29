import os
from typing import List, Optional

def get_start_epoch(cfg) -> int:

    if not cfg.initialize.resume:
        return 0
    
    # Try to get epoch from checkpoint file first
    ckpt_path = os.path.join(cfg.PATH.CHECKPOINT_ROOT, cfg.initialize.run_id, cfg.initialize.checkpoint)
    next_epoch = 1 if 'epoch' in cfg.initialize.checkpoint else 0  # +1 because we want to start from the next epoch

    import torch
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False, mmap=True)
    start_epoch = int(checkpoint['epoch']) + next_epoch 
    print(f"Resuming from epoch {start_epoch} (from checkpoint)")
    
    return start_epoch

def get_profiler(checkpoint_path: str):
    from torch.profiler import schedule, tensorboard_trace_handler
    from lightning.pytorch.profilers import PyTorchProfiler
    
    pl_profiler = PyTorchProfiler(
        dirpath=os.path.join(checkpoint_path, 'profiler'), 
        filename='profiler',
        # on_trace_ready=tensorboard_trace_handler(os.path.join(CHECKPOINT_PATH, 'profiler')), # Use this only for tensorboard
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        schedule = schedule(skip_first=170, wait=10, warmup=10, active=20, repeat=1),
        row_limit = -1,
        sort_by_key="cuda_time",
        # export_to_chrome=False
        ) 
    return pl_profiler

def copy_files(src_path: str, dst_path: str, filenames: List[str], compare_files: bool = False):
    import filecmp
    import shutil
    
    print(f'Copying files to directory...')
    if not os.path.exists(dst_path):
        os.makedirs(dst_path, exist_ok=True)
    copy_count = 0
    for file in filenames:
        src_file = os.path.join(src_path, file)
        dst_file = os.path.join(dst_path, file)
        if not os.path.exists(dst_file) or (compare_files and not filecmp.cmp(src_file, dst_file)):
            shutil.copy(src_file, dst_file)
            copy_count += 1
    print(f'{copy_count} files copied successfully!')
    