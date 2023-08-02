# experiment 2.1

### TLDR

trying to see if hanging would go away with `CUDA_LAUNCH_BLOCKING=1` and also using https://github.com/huggingface/m4/pull/1111 to see if context manager used by accelerate to do grad accum is an issue - as we discovered that contextlib suppressed some exceptions - and that's what troubles us - we get hanging, but we get no tracebacks that tell us where the problem is.

It didn't help.

Though this result is inconclusive as this run coincided with multiple JZ meltdowns, so we don't know if the failure was related to the latter.


### Setup

```
cd /gpfsdswork/projects/rech/cnw/commun/experiments/stas/m4-no-ctx-mgr
bash experiments/pretraining/vloom/tr_141_cm409xPMD01_scale_leap_of_faith_v5_num_workers_02/01_launch.sh

cd /gpfsssd/scratch/rech/cnw/commun/experiments/local_experiment_dir/tr_141_cm409xPMD01_scale_leap_of_faith_v5_num_workers_02/logs
tail -f main_log.txt
```

setup:

- no accumulate context manager PR#1111
- pt-1.12
- CUDA_LAUNCH_BLOCKING=1
- grad_clip: 1.0
- num_workers=2


last good debug log:
```
PID 2232678:  After save batch at opt_step 833
```

log:

```
terminate called after throwing an instance of 'terminate called after throwing an instance of 'c10::CUDAError'
c10::CUDAError'
  what():  CUDA error: unknown error
Exception raised from query at ../aten/src/ATen/cuda/CUDAEvent.h:91 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x3e (0x147b59a2c20e in /gpfswork/rech/cnw/commun/conda/shared-m4-2023-02-28/lib/python3.8/site-packages/torch/lib/libc10.so)
frame #1: c10d::ProcessGroupNCCL::WorkNCCL::finishedGPUExecutionInternal() const + 0x13c (0x147b9b82c78c in /gpfswork/rech/cnw/commun/conda/shared-m4-2023-02-28/lib/python3.8/site-packages/torch/lib/libtorch_cuda_cpp.so)
frame #2: c10d::ProcessGroupNCCL::WorkNCCL::isCompleted() + 0x58 (0x147b9b82e768 in /gpfswork/rech/cnw/commun/conda/shared-m4-2023-02-28/lib/python3.8/site-packages/torch/lib/libtorch_cuda_cpp.so)
frame #3: c10d::ProcessGroupNCCL::workCleanupLoop() + 0x221 (0x147b9b82fcf1 in /gpfswork/rech/cnw/commun/conda/shared-m4-2023-02-28/lib/python3.8/site-packages/torch/lib/libtorch_cuda_cpp.so)
frame #4: <unknown function> + 0xc2ba3 (0x147bab5b7ba3 in /lib64/libstdc++.so.6)
frame #5: <unknown function> + 0x81cf (0x147bca0531cf in /lib64/libpthread.so.0)
frame #6: clone + 0x43 (0x147bc9535dd3 in /lib64/libc.so.6)
  what():  CUDA error: unknown error
Exception raised from query at ../aten/src/ATen/cuda/CUDAEvent.h:91 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x3e (0x14c5bad6d20e in /gpfswork/rech/cnw/commun/conda/shared-m4-2023-02-28/lib/python3.8/site-packages/torch/lib/libc10.so)
frame #1: c10d::ProcessGroupNCCL::WorkNCCL::finishedGPUExecutionInternal() const + 0x13c (0x14c5fcb6d78c in /gpfswork/rech/cnw/commun/conda/shared-m4-2023-02-28/lib/python3.8/site-packages/torch/lib/libtorch_cuda_cpp.so)
frame #2: c10d::ProcessGroupNCCL::WorkNCCL::isCompleted() + 0x58 (0x14c5fcb6f768 in /gpfswork/rech/cnw/commun/conda/shared-m4-2023-02-28/lib/python3.8/site-packages/torch/lib/libtorch_cuda_cpp.so)
frame #3: c10d::ProcessGroupNCCL::workCleanupLoop() + 0x221 (0x14c5fcb70cf1 in /gpfswork/rech/cnw/commun/conda/shared-m4-2023-02-28/lib/python3.8/site-packages/torch/lib/libtorch_cuda_cpp.so)
frame #4: <unknown function> + 0xc2ba3 (0x14c60c8f8ba3 in /lib64/libstdc++.so.6)
frame #5: <unknown function> + 0x81cf (0x14c62b3941cf in /lib64/libpthread.so.0)
frame #6: clone + 0x43 (0x14c62a876dd3 in /lib64/libc.so.6)


terminate called after throwing an instance of 'c10::CUDAError'
  what():  CUDA error: unknown error
Exception raised from query at ../aten/src/ATen/cuda/CUDAEvent.h:91 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x3e (0x14c5bad6d20e in /gpfswork/rech/cnw/commun/conda/shared-m4-2023-02-28/lib/python3.8/site-packages/torch/lib/libc10.so)
frame #1: c10d::ProcessGroupNCCL::WorkNCCL::finishedGPUExecutionInternal() const + 0x13c (0x14c5fcb6d78c in /gpfswork/rech/cnw/commun/conda/shared-m4-2023-02-28/lib/python3.8/site-packages/torch/lib/libtorch_cuda_cpp.so)
frame #2: c10d::ProcessGroupNCCL::WorkNCCL::isCompleted() + 0x58 (0x14c5fcb6f768 in /gpfswork/rech/cnw/commun/conda/shared-m4-2023-02-28/lib/python3.8/site-packages/torch/lib/libtorch_cuda_cpp.so)
frame #3: c10d::ProcessGroupNCCL::workCleanupLoop() + 0x221 (0x14c5fcb70cf1 in /gpfswork/rech/cnw/commun/conda/shared-m4-2023-02-28/lib/python3.8/site-packages/torch/lib/libtorch_cuda_cpp.so)
frame #4: <unknown function> + 0xc2ba3 (0x14c60c8f8ba3 in /lib64/libstdc++.so.6)
frame #5: <unknown function> + 0x81cf (0x14c62b3941cf in /lib64/libpthread.so.0)
frame #6: clone + 0x43 (0x14c62a876dd3 in /lib64/libc.so.6)

srun: error: Node failure on jean-zay-iam49
srun: Job step aborted: Waiting up to 62 seconds for job step to finish.
```
