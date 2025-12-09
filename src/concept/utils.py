import os

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
        schedule = schedule(skip_first=100, wait=10, warmup=10, active=20, repeat=1),
        row_limit = -1,
        sort_by_key="cuda_time",
        # export_to_chrome=False
        ) 
    return pl_profiler