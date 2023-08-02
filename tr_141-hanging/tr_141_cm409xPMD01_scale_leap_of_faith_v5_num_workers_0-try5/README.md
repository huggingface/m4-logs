# experiment 0.5


### TLDR

Using https://github.com/huggingface/m4/pull/1111 to see if context manager used by accelerate to do grad accum is an issue
and trying `grad_clip=0.0`

Training stopped with one whole node stopping to respond at all - couldn't ssh to it.

This result is inconclusive as this run coincided with multiple JZ meltdowns, so we don't know if the failure was related to the latter.


### Setup
```
cd /gpfsdswork/projects/rech/cnw/commun/experiments/stas/m4-no-ctx-mgr
tr_141_cm409xPMD01_scale_leap_of_faith_v5_num_workers_0-try5
```

- no accumulate context manager PR#1111
- pt-1.12
- CUDA_LAUNCH_BLOCKING=1
- grad_clip: 0.0﻿
- num_workers = 0




### Investigation


as it was still hanging I was able to get `py-spy` dumps

```
srun --overlap --jobid=1911774 --gres=gpu:0 --nodes=8 --tasks-per-node=1 --output=trace-%N.out bash -c 'source $cnw_ALL_CCFRWORK/start-m4-user; conda activate stas-m4; pgrep -P $(pgrep -o accelerate) | xargs -I {} py-spy dump --pid {} || echo "failed"; sleep 3'
```

node 15 didn't generate the log and isn't responding

```
ssh jean-zay-iam15
```
won't connect, but can ping:
```
ping jean-zay-iam15
PING jean-zay-iam15-ib0 (10.148.8.232) 56(84) bytes of data.
64 bytes from jean-zay-iam15-ib0 (10.148.8.232): icmp_seq=1 ttl=64 time=0.100 ms
64 bytes from jean-zay-iam15-ib0 (10.148.8.232): icmp_seq=2 ttl=64 time=0.115 ms
```
tried again to ssh, got:
```
ssh jean-zay-iam15
Last login: Sat Feb  4 05:04:51 2023 from 10.148.0.20
```
but no further.


end of main_log.txt
```
PID 431169:  After activation tracker at opt_step 235
[...]
PID 431171:  Before backward pass in _do_batch
PID 431170:  After loss computation in _do_batch
PID 431170:  Before backward pass in _do_batch
PID 431169:  After loss computation in _do_batch
PID 431169:  Before backward pass in _do_batch
terminate called after throwing an instance of 'c10::CUDAError'
terminate called after throwing an instance of 'c10::CUDAError'
  what():  CUDA error: unknown error
Exception raised from query at ../aten/src/ATen/cuda/CUDAEvent.h:91 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x3e (0x148d35fe220e in /gpfswork/rech/cnw/commun/conda/shared-m4-2023-02-28/lib/python3.8/site-packages/torch/lib/libc10.so)
frame #1: c10d::ProcessGroupNCCL::WorkNCCL::finishedGPUExecutionInternal() const + 0x13c (0x148d77de278c in /gpfswork/rech/cnw/commun/conda/shared-m4-2023-02-28/lib/python3.8/site-packages/torch/lib/libtorch_cuda_cpp.so)
frame #2: c10d::ProcessGroupNCCL::WorkNCCL::isCompleted() + 0x58 (0x148d77de4768 in /gpfswork/rech/cnw/commun/conda/shared-m4-2023-02-28/lib/python3.8/site-packages/torch/lib/libtorch_cuda_cpp.so)
frame #3: c10d::ProcessGroupNCCL::workCleanupLoop() + 0x221 (0x148d77de5cf1 in /gpfswork/rech/cnw/commun/conda/shared-m4-2023-02-28/lib/python3.8/site-packages/torch/lib/libtorch_cuda_cpp.so)
frame #4: <unknown function> + 0xc2ba3 (0x148d87b6dba3 in /lib64/libstdc++.so.6)
frame #5: <unknown function> + 0x81cf (0x148da66091cf in /lib64/libpthread.so.0)
frame #6: clone + 0x43 (0x148da5aebdd3 in /lib64/libc.so.6)

  what():  CUDA error: unknown error
Exception raised from query at ../aten/src/ATen/cuda/CUDAEvent.h:91 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x3e (0x153a17a5720e in /gpfswork/rech/cnw/commun/conda/shared-m4-2023-02-28/lib/python3.8/site-packages/torch/lib/libc10.so)
frame #1: c10d::ProcessGroupNCCL::WorkNCCL::finishedGPUExecutionInternal() const + 0x13c (0x153a5985778c in /gpfswork/rech/cnw/commun/conda/shared-m4-2023-02-28/lib/python3.8/site-packages/torch/lib/libtorch_cuda_cpp.so)
frame #2: c10d::ProcessGroupNCCL::WorkNCCL::isCompleted() + 0x58 (0x153a59859768 in /gpfswork/rech/cnw/commun/conda/shared-m4-2023-02-28/lib/python3.8/site-packages/torch/lib/libtorch_cuda_cpp.so)
frame #3: c10d::ProcessGroupNCCL::workCleanupLoop() + 0x221 (0x153a5985acf1 in /gpfswork/rech/cnw/commun/conda/shared-m4-2023-02-28/lib/python3.8/site-packages/torch/lib/libtorch_cuda_cpp.so)
frame #4: <unknown function> + 0xc2ba3 (0x153a695e2ba3 in /lib64/libstdc++.so.6)
frame #5: <unknown function> + 0x81cf (0x153a8807e1cf in /lib64/libpthread.so.0)
frame #6: clone + 0x43 (0x153a87560dd3 in /lib64/libc.so.6)
```

Trying to see if there were any cores:
```
cd /gpfsdswork/projects/rech/cnw/commun/experiments/stas/m4-no-ctx-mgr
ls -l
```
was very slow to respond - many core files!
```
-rw-------  1 ura81os cnw  434M Mar  4 19:10 core-accelerate-177773-7
-rw-------  1 ura81os cnw  370M Mar  4 19:10 core-accelerate-177877-7
-rw-------  1 ura81os cnw  370M Mar  4 19:10 core-accelerate-2087854-7
-rw-------  1 ura81os cnw  371M Mar  4 19:10 core-accelerate-352909-7
-rw-------  1 ura81os cnw  370M Mar  4 19:10 core-accelerate-582000-7
-rw-------  1 ura81os cnw  370M Mar  4 19:10 core-accelerate-635804-7
-rw-------  1 ura81os cnw  370M Mar  4 19:10 core-accelerate-736384-7
-rw-------  1 ura81os cnw  370M Mar  4 19:10 core-accelerate-956281-7
-rw-------  1 ura81os cnw  4.0G Mar  4 19:10 core-python-177863-7
-rw-------  1 ura81os cnw  3.1G Mar  4 19:10 core-python-177864-7
-rw-------  1 ura81os cnw  3.1G Mar  4 19:10 core-python-177865-7
-rw-------  1 ura81os cnw  2.6G Mar  4 19:10 core-python-177866-7
-rw-------  1 ura81os cnw  3.0G Mar  4 19:10 core-python-177867-7
-rw-------  1 ura81os cnw  2.6G Mar  4 19:10 core-python-177868-7
-rw-------  1 ura81os cnw  3.0G Mar  4 19:10 core-python-177869-7
-rw-------  1 ura81os cnw  3.1G Mar  4 19:10 core-python-177870-7
-rw-------  1 ura81os cnw  2.7G Mar  4 19:10 core-python-177967-7
-rw-------  1 ura81os cnw  2.9G Mar  4 19:10 core-python-177968-7
-rw-------  1 ura81os cnw  2.8G Mar  4 19:10 core-python-177969-7
-rw-------  1 ura81os cnw  2.8G Mar  4 19:10 core-python-177970-7
-rw-------  1 ura81os cnw  2.9G Mar  4 19:10 core-python-177971-7
-rw-------  1 ura81os cnw  2.8G Mar  4 19:10 core-python-177972-7
-rw-------  1 ura81os cnw  2.8G Mar  4 19:10 core-python-177973-7
-rw-------  1 ura81os cnw  2.7G Mar  4 19:10 core-python-177974-7
-rw-------  1 ura81os cnw  1.3G Mar  4 19:10 core-python-2087943-7
-rw-------  1 ura81os cnw  1.1G Mar  4 19:10 core-python-2087944-7
-rw-------  1 ura81os cnw  1.1G Mar  4 19:10 core-python-2087945-7
-rw-------  1 ura81os cnw  1.1G Mar  4 19:10 core-python-2087946-7
-rw-------  1 ura81os cnw  1.1G Mar  4 19:10 core-python-2087947-7
-rw-------  1 ura81os cnw  1.1G Mar  4 19:10 core-python-2087948-7
-rw-------  1 ura81os cnw  1.1G Mar  4 19:10 core-python-2087949-7
-rw-------  1 ura81os cnw  1.1G Mar  4 19:10 core-python-2087950-7
-rw-------  1 ura81os cnw   10G Mar  4 00:24 core-python-273548-6
-rw-------  1 ura81os cnw  973M Mar  4 19:10 core-python-353001-7
-rw-------  1 ura81os cnw  783M Mar  4 19:10 core-python-353002-7
-rw-------  1 ura81os cnw  793M Mar  4 19:10 core-python-353003-7
-rw-------  1 ura81os cnw  871M Mar  4 19:10 core-python-353004-7
-rw-------  1 ura81os cnw  851M Mar  4 19:10 core-python-353005-7
-rw-------  1 ura81os cnw  845M Mar  4 19:10 core-python-353006-7
-rw-------  1 ura81os cnw  836M Mar  4 19:10 core-python-353007-7
-rw-------  1 ura81os cnw  828M Mar  4 19:10 core-python-353008-7
-rw-------  1 ura81os cnw  8.4G Mar  5 04:35 core-python-527195-6
-rw-------  1 ura81os cnw  8.3G Mar  5 04:36 core-python-527196-6
-rw-------  1 ura81os cnw  1.1G Mar  4 19:10 core-python-582091-7
-rw-------  1 ura81os cnw  776M Mar  4 19:10 core-python-582092-7
-rw-------  1 ura81os cnw  1.1G Mar  4 19:10 core-python-582093-7
-rw-------  1 ura81os cnw  1.1G Mar  4 19:10 core-python-582094-7
-rw-------  1 ura81os cnw  1.1G Mar  4 19:10 core-python-582095-7
-rw-------  1 ura81os cnw  1.1G Mar  4 19:10 core-python-582096-7
-rw-------  1 ura81os cnw 1017M Mar  4 19:10 core-python-582097-7
-rw-------  1 ura81os cnw  1.1G Mar  4 19:10 core-python-582098-7
-rw-------  1 ura81os cnw  1.3G Mar  4 19:10 core-python-635895-7
-rw-------  1 ura81os cnw  1.3G Mar  4 19:10 core-python-635896-7
-rw-------  1 ura81os cnw  1.3G Mar  4 19:10 core-python-635897-7
-rw-------  1 ura81os cnw  1.3G Mar  4 19:10 core-python-635898-7
-rw-------  1 ura81os cnw  1.2G Mar  4 19:10 core-python-635899-7
-rw-------  1 ura81os cnw  1.3G Mar  4 19:10 core-python-635900-7
-rw-------  1 ura81os cnw  1.3G Mar  4 19:10 core-python-635901-7
-rw-------  1 ura81os cnw  1.3G Mar  4 19:10 core-python-635902-7
-rw-------  1 ura81os cnw  1.5G Mar  4 19:10 core-python-736475-7
-rw-------  1 ura81os cnw  1.3G Mar  4 19:10 core-python-736476-7
-rw-------  1 ura81os cnw  1.2G Mar  4 19:10 core-python-736477-7
-rw-------  1 ura81os cnw  1.2G Mar  4 19:10 core-python-736478-7
-rw-------  1 ura81os cnw  1.2G Mar  4 19:10 core-python-736479-7
-rw-------  1 ura81os cnw  1.2G Mar  4 19:10 core-python-736480-7
-rw-------  1 ura81os cnw  1.2G Mar  4 19:10 core-python-736481-7
-rw-------  1 ura81os cnw  1.2G Mar  4 19:10 core-python-736482-7
-rw-------  1 ura81os cnw  1.2G Mar  4 19:10 core-python-956372-7
-rw-------  1 ura81os cnw  1.1G Mar  4 19:10 core-python-956373-7
-rw-------  1 ura81os cnw 1023M Mar  4 19:10 core-python-956374-7
-rw-------  1 ura81os cnw  1.1G Mar  4 19:10 core-python-956375-7
-rw-------  1 ura81os cnw  1.1G Mar  4 19:10 core-python-956376-7
-rw-------  1 ura81os cnw  1.1G Mar  4 19:10 core-python-956377-7
-rw-------  1 ura81os cnw  997M Mar  4 19:10 core-python-956379-7
-rw-------  1 ura81os cnw  1.1G Mar  4 19:10 core-python-956380-7
-rw-------  1 ura81os cnw   17M Mar  4 19:10 core-srun-352887-6
``````
```

Tried to get bt from a few

```
gdb /gpfswork/rech/cnw/commun/conda/shared-m4-2023-02-28/bin/python core-python-353004-7

(gdb) bt
#0  0x0000154fc2d16d18 in ?? ()
Backtrace stopped: Cannot access memory at address 0x154f6550edc0
```

tried a few many seem like that. I didn't find any corefiles that weren't broken.


rm'ed core files as there were too many and all useless
