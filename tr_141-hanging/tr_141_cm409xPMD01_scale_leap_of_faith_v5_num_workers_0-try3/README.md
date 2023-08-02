# experiment 3.1

## TLDR

64 gpus, node0:rank0 goes to deep sleep after 3.5h of training and won't respond. gpu0 seems to be totally fine - can query, low temp, etc. nothing in kernel logs.

It points to a software problem since it happens again on a new set of nodes. Trying to find the trigger.

Apparently the HPC had a meltdown - the filesystem froze and became unresponsive, node0:rank0 was trying to log and as the system call never returned, so it went into a deep sleep and the whole thing froze.

c10d won't even abort after 30min timeout.


### Setup

```
cd /gpfsdswork/projects/rech/cnw/commun/experiments/stas/m4-full
bash experiments/pretraining/vloom/tr_141_cm409xPMD01_scale_leap_of_faith_v5_num_workers_03/01_launch.sh
cd /gpfsssd/scratch/rech/cnw/commun/experiments/local_experiment_dir/tr_141_cm409xPMD01_scale_leap_of_faith_v5_num_workers_03/logs
tail -f main_log.txt
```

setup:

- m4@main - no tweaks
- pt-1.12
- grad_clip: 1.0
- CUDA_LAUNCH_BLOCKING=1
- num_workers=2
- normal accumulation (really using m4@main)

failed to start with:

```
    return cls.__new__(cls, value)
  File "/gpfswork/rech/cnw/commun/conda/shared-m4-2023-02-28/lib/python3.8/enum.py", line 663, in __new__
    raise ve_exc
ValueError: 'jsonl' is not a valid LoggingTypes
```

so had to remove `-jsonl` from config in 2 places


### Investigation

As c10d was hanging for indefinite time I had the luxury to study the situation at my pace

I notice that the main log stopped writing

280058 is our hanging process - how I found it will write later

the last few lines of the log are:
```
PID 756282:  After clip in _do_batch
PID 756284:  Before sync_gradients in _do_batch
PID 280065:  Before sync_gradients in _do_batch
PID 280061:  Before sync_gradients in _do_batch
PID 515584:  After clip in _do_batch
PID 280062:  After deepspeed backward pass in _do_batch
PID 280059:  Before sync_gradients in _do_batch
PID 756286:  After clip in _do_batch
PID 515583:  After clip in _do_batch
PID 280060:  After deepspeed backward pass in _do_batch
PID 78352:  After clip in _do_batch
PID 78358:  After clip in _do_batch
PID 1966885:  After clip in _do_batch
PID 280064:  Before sync_gradients in _do_batch
PID 280065:  After clip in _do_batch
PID 1966887:  Before sync_gradients in _do_batch
PID 756284:  After clip in _do_batch
PID 1966887:  After clip in _do_batch
PID 280061:  After clip in _do_batch
PID 280064:  After clip in _do_batch
PID 280059:  After clip in _do_batch
PID 280058:  Before sync_gradients in _do_batch
PID 280060:  Before sync_gradients in _do_batch
PID 280062:  Before sync_gradients in _do_batch
PID 280058:  After clip in _do_batch
PID 280060:  After clip in _do_batch
PID 280062:  After clip in _do_batch
```
So we know that grad clipping was the last thing to run

ok, I was lucky to be around, so I started investigating. let's get `py-spy`
```
srun --overlap --jobid=1882409 --gres=gpu:0 --nodes=8 --tasks-per-node=1 --output=trace-%N.out bash -c 'source $cnw_ALL_CCFRWORK/start-m4-user; conda activate stas-m4; pgrep -P $(pgrep -o accelerate) | xargs -I {} py-spy dump --pid {}' || echo "failed"srun: error: jean-zay-iam33: task 3: Exited with exit code 123
srun: launch/slurm: _step_signal: Terminating StepId=1882409.5
srun: error: jean-zay-iam06: task 0: Terminated
srun: error: jean-zay-iam31: task 1: Terminated
srun: error: jean-zay-iam39: task 4: Terminated
srun: error: jean-zay-iam32: task 2: Terminated
srun: error: jean-zay-iam50: task 5: Terminated
srun: error: jean-zay-iam51: task 6: Terminated
srun: error: jean-zay-iam52: task 7: Terminated
srun: Force Terminated StepId=1882409.5
failed
```

I checked it looked like every process that did report was in `barrier`, e.g.:
```
Thread 280064 (active): "MainThread"
    barrier (torch/distributed/distributed_c10d.py:2784)
    wait_for_everyone (accelerate/utils/other.py:81)
    wait_for_everyone (accelerate/accelerator.py:1747)
```
I used this command to get all processes at once:
```
grep -A2 MainT trace-jean-zay-iam*
```
so I counted how many of them were in barrier and got:
```
find . -name "trace*" -exec sh -c 'echo "$1: $(grep "barrier (torch/distri" $1 | wc -l)"' _ {} \;
./trace-jean-zay-iam32.out: 7
./trace-jean-zay-iam52.out: 6
./trace-jean-zay-iam06.out: 7
./trace-jean-zay-iam31.out: 6
./trace-jean-zay-iam33.out: 6
./trace-jean-zay-iam51.out: 7
./trace-jean-zay-iam39.out: 7
./trace-jean-zay-iam50.out: 6
```
weird, missing reports, I added sleep 3, after py-spy and all got 8 matches on all nodes except 06 - so I knew it was the problematic node.
```
srun --overlap --jobid=1882409 --gres=gpu:0 --nodes=8 --tasks-per-node=1 --output=trace-%N.out bash -c \
'source $cnw_ALL_CCFRWORK/start-m4-user; conda activate stas-m4; pgrep -P $(pgrep -o accelerate) | \
xargs -I {} py-spy dump --pid {}; sleep 3' || echo "failed"
```
ok going into the problematic node
```
ssh jean-zay-iam06
source $cnw_ALL_CCFRWORK/start-m4-user; conda activate stas-m4;
pgrep -P $(pgrep -o accelerate) | xargs -I {} py-spy dump --pid {} > trace-2/trace-jean-zay-iam06-2.out
Error: Failed to suspend process
Reason: EPERM: Operation not permitted
Reason: EPERM: Operation not permitted
```
I discover gpu0 isn't in the report, that was the error above.

I then look at nvidia status and see that all gpus  are at 100%, except node0:rank0 is at 0%
```
nvidia-smi --format=csv --query-gpu=power.draw,utilization.gpu,fan.speed,temperature.gpu
power.draw [W], utilization.gpu [%], fan.speed [%], temperature.gpu
67.97 W, 0 %, [N/A], 28
87.12 W, 100 %, [N/A], 30
84.83 W, 100 %, [N/A], 28
85.09 W, 100 %, [N/A], 32
86.67 W, 100 %, [N/A], 30
84.99 W, 100 %, [N/A], 31
85.15 W, 100 %, [N/A], 30
87.26 W, 100 %, [N/A], 32
```
longer report of the same:
```
nvidia-smi

Sat Mar  4 04:38:36 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.60.13    Driver Version: 525.60.13    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  On   | 00000000:07:00.0 Off |                    0 |
| N/A   28C    P0    68W / 400W |  22377MiB / 81920MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA A100-SXM...  On   | 00000000:0B:00.0 Off |                    0 |
| N/A   30C    P0    87W / 400W |  15995MiB / 81920MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA A100-SXM...  On   | 00000000:48:00.0 Off |                    0 |
| N/A   28C    P0    84W / 400W |  16122MiB / 81920MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA A100-SXM...  On   | 00000000:4C:00.0 Off |                    0 |
| N/A   31C    P0    85W / 400W |  15995MiB / 81920MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   4  NVIDIA A100-SXM...  On   | 00000000:88:00.0 Off |                    0 |
| N/A   29C    P0    86W / 400W |  15995MiB / 81920MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   5  NVIDIA A100-SXM...  On   | 00000000:8B:00.0 Off |                    0 |
| N/A   31C    P0    84W / 400W |  15995MiB / 81920MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   6  NVIDIA A100-SXM...  On   | 00000000:C8:00.0 Off |                    0 |
| N/A   30C    P0    85W / 400W |  15995MiB / 81920MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   7  NVIDIA A100-SXM...  On   | 00000000:CB:00.0 Off |                    0 |
| N/A   32C    P0    87W / 400W |  15947MiB / 81920MiB |    100%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A    280058      C   ...-m4-2023-02-28/bin/python    22374MiB |
|    1   N/A  N/A    280059      C   ...-m4-2023-02-28/bin/python    15992MiB |
|    2   N/A  N/A     19501      G   /usr/libexec/Xorg                  63MiB |
|    2   N/A  N/A     20379      G   /usr/bin/gnome-shell               63MiB |
|    2   N/A  N/A    280060      C   ...-m4-2023-02-28/bin/python    15992MiB |
|    3   N/A  N/A    280061      C   ...-m4-2023-02-28/bin/python    15992MiB |
|    4   N/A  N/A    280062      C   ...-m4-2023-02-28/bin/python    15992MiB |
|    5   N/A  N/A    280063      C   ...-m4-2023-02-28/bin/python    15992MiB |
|    6   N/A  N/A    280064      C   ...-m4-2023-02-28/bin/python    15992MiB |
|    7   N/A  N/A    280065      C   ...-m4-2023-02-28/bin/python    15944MiB |
+-----------------------------------------------------------------------------+
```

alright, so we can now see from nvidia-smi that pid 280059 is our problematic process. Because it corresponds to gpu0.
```
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A    280058      C   ...-m4-2023-02-28/bin/python    22374MiB |
```

Let's try to strace a good process:
```
strace -p 280059
strace: Process 280059 attached
[ Process PID=280059 runs in x32 mode. ]
[ Process PID=280059 runs in 64 bit mode. ]
ioctl(3, _IOC(_IOC_READ|_IOC_WRITE, 0x46, 0x2a, 0x20), 0x7fff748d1d60) = 0
ioctl(3, _IOC(_IOC_READ|_IOC_WRITE, 0x46, 0x2a, 0x20), 0x7fff748d1d60) = 0
ioctl(3, _IOC(_IOC_READ|_IOC_WRITE, 0x46, 0x2a, 0x20), 0x7fff748d1d60) = 0
ioctl(3, _IOC(_IOC_READ|_IOC_WRITE, 0x46, 0x2a, 0x20), 0x7fff748d1d60) = 0
ioctl(3, _IOC(_IOC_READ|_IOC_WRITE, 0x46, 0x2a, 0x20), 0x7fff748d1d60) = 0
^Cstrace: Process 280059 detached
```
looking good - matching the barrier - we can see it's doing polling on fd0 which is nvidiactl device via
```
ls -l /proc/280059/fd/3
```
But if we try to attach to the problematic process we get:
```
strace -p 280058
strace: attach: ptrace(PTRACE_SEIZE, 280058): Operation not permitted
```

and py-spy of course fails too:
```
py-spy dump --pid 280058
Process 280058: ...
Error: Failed to suspend process
Reason: EPERM: Operation not permitted
Reason: EPERM: Operation not permitted
```

Let's look at process status next:
```
ps auxc | head -1; ps auxc | grep python
USER         PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
ura81os   280058  117  4.6 17517200480 24494736 ? Dl  00:36 291:14 python
ura81os   280059 87.9  4.4 17508566232 23526964 ? Rl  00:36 217:52 python
ura81os   280060 87.7  5.1 17512092688 27323380 ? Rl  00:36 217:12 python
ura81os   280061 88.4  4.2 17508590204 22350536 ? Rl  00:36 219:04 python
ura81os   280062 88.0  4.8 17510468232 25509144 ? Rl  00:36 218:03 python
ura81os   280063 88.1  4.3 17508576736 22844944 ? Rl  00:36 218:13 python
ura81os   280064 87.9  4.3 17508567660 22901892 ? Rl  00:36 217:52 python
ura81os   280065  105  4.5 17510297420 24196752 ? Rl  00:36 262:09 python
```
we can see it's the 0th gpu's process of the master node that is in deep sleep
```
Dl = uninterruptible sleep (usually IO) / is multi-threaded
Rl = running or runnable (on run queue) / is multi-threaded
```
here is the decyhpering table for the STAT column:
```
PROCESS STATE CODES
       Here are the different values that the s, stat and state output specifiers (header "STAT" or "S") will display to describe the state of a process:
       D    uninterruptible sleep (usually IO)
       R    running or runnable (on run queue)
       S    interruptible sleep (waiting for an event to complete)
       T    stopped, either by a job control signal or because it is being traced.
       W    paging (not valid since the 2.6.xx kernel)
       X    dead (should never be seen)
       Z    defunct ("zombie") process, terminated but not reaped by its parent.

       For BSD formats and when the stat keyword is used, additional characters may be displayed:
       <    high-priority (not nice to other users)
       N    low-priority (nice to other users)
       L    has pages locked into memory (for real-time and custom IO)
       s    is a session leader
       l    is multi-threaded (using CLONE_THREAD, like NPTL pthreads do)
       +    is in the foreground process group.
```

ok, so the process got borked and there is nothing can be done other than rebooting the node.

The gpu though appears to be totally fine:
```
nvidia-smi -q -d temperature

==============NVSMI LOG==============

Timestamp                                 : Sat Mar  4 04:46:50 2023
Driver Version                            : 525.60.13
CUDA Version                              : 12.0

Attached GPUs                             : 8
GPU 00000000:07:00.0
    Temperature
        GPU Current Temp                  : 28 C
        GPU Shutdown Temp                 : 92 C
        GPU Slowdown Temp                 : 89 C
        GPU Max Operating Temp            : 85 C
        GPU Target Temperature            : N/A
        Memory Current Temp               : 45 C
        Memory Max Operating Temp         : 95 C

GPU 00000000:0B:00.0
    Temperature
        GPU Current Temp                  : 30 C
        GPU Shutdown Temp                 : 92 C
        GPU Slowdown Temp                 : 89 C
        GPU Max Operating Temp            : 85 C
        GPU Target Temperature            : N/A
        Memory Current Temp               : 43 C
        Memory Max Operating Temp         : 95 C
[...]
```
here is the shorter version w/ all gpus - all looks good
```
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader
28
30
28
31
29
31
29
32
```


# query gpu 0 for all of its stats
```
nvidia-smi -q -i 0

==============NVSMI LOG==============

Timestamp                                 : Sat Mar  4 04:55:25 2023
Driver Version                            : 525.60.13
CUDA Version                              : 12.0

Attached GPUs                             : 8
GPU 00000000:07:00.0
    Product Name                          : NVIDIA A100-SXM4-80GB
    Product Brand                         : NVIDIA
    Product Architecture                  : Ampere
    Display Mode                          : Enabled
    Display Active                        : Disabled
    Persistence Mode                      : Enabled
    MIG Mode
        Current                           : Disabled
        Pending                           : Disabled
    Accounting Mode                       : Disabled
    Accounting Mode Buffer Size           : 4000
    Driver Model
        Current                           : N/A
        Pending                           : N/A
    Serial Number                         : 1562221013740
    GPU UUID                              : GPU-eb0003d2-77b4-70fd-4ab9-3ef1c9853c49
    Minor Number                          : 2
    VBIOS Version                         : 92.00.45.00.05
    MultiGPU Board                        : No
    Board ID                              : 0x700
    Board Part Number                     : 692-2G506-0210-002
    GPU Part Number                       : 20B2-895-A1
    Module ID                             : 3
    Inforom Version
        Image Version                     : G506.0210.00.03
        OEM Object                        : 2.0
        ECC Object                        : 6.16
        Power Management Object           : N/A
    GPU Operation Mode
        Current                           : N/A
        Pending                           : N/A
    GSP Firmware Version                  : 525.60.13
    GPU Virtualization Mode
        Virtualization Mode               : None
        Host VGPU Mode                    : N/A
    IBMNPU
        Relaxed Ordering Mode             : N/A
    PCI
        Bus                               : 0x07
        Device                            : 0x00
        Domain                            : 0x0000
        Device Id                         : 0x20B210DE
        Bus Id                            : 00000000:07:00.0
        Sub System Id                     : 0x146310DE
        GPU Link Info
            PCIe Generation
                Max                       : 4
                Current                   : 4
                Device Current            : 4
                Device Max                : 4
                Host Max                  : 4
            Link Width
                Max                       : 16x
                Current                   : 16x
        Bridge Chip
            Type                          : N/A
            Firmware                      : N/A
        Replays Since Reset               : 0
        Replay Number Rollovers           : 0
        Tx Throughput                     : 0 KB/s
        Rx Throughput                     : 0 KB/s
        Atomic Caps Inbound               : N/A
        Atomic Caps Outbound              : N/A
    Fan Speed                             : N/A
    Performance State                     : P0
    Clocks Throttle Reasons
        Idle                              : Not Active
        Applications Clocks Setting       : Not Active
        SW Power Cap                      : Not Active
        HW Slowdown                       : Not Active
            HW Thermal Slowdown           : Not Active
            HW Power Brake Slowdown       : Not Active
        Sync Boost                        : Not Active
        SW Thermal Slowdown               : Not Active
        Display Clock Setting             : Not Active
    FB Memory Usage
        Total                             : 81920 MiB
        Reserved                          : 834 MiB
        Used                              : 22377 MiB
        Free                              : 58708 MiB
    BAR1 Memory Usage
        Total                             : 131072 MiB
        Used                              : 4 MiB
        Free                              : 131068 MiB
    Compute Mode                          : Default
    Utilization
        Gpu                               : 0 %
        Memory                            : 0 %
        Encoder                           : 0 %
        Decoder                           : 0 %
    Encoder Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    FBC Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    Ecc Mode
        Current                           : Enabled
        Pending                           : Enabled
    ECC Errors
        Volatile
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
        Aggregate
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
    Retired Pages
        Single Bit ECC                    : N/A
        Double Bit ECC                    : N/A
        Pending Page Blacklist            : N/A
    Remapped Rows
        Correctable Error                 : 0
        Uncorrectable Error               : 0
        Pending                           : No
        Remapping Failure Occurred        : No
        Bank Remap Availability Histogram
            Max                           : 640 bank(s)
            High                          : 0 bank(s)
            Partial                       : 0 bank(s)
            Low                           : 0 bank(s)
            None                          : 0 bank(s)
    Temperature
        GPU Current Temp                  : 28 C
        GPU Shutdown Temp                 : 92 C
        GPU Slowdown Temp                 : 89 C
        GPU Max Operating Temp            : 85 C
        GPU Target Temperature            : N/A
        Memory Current Temp               : 46 C
        Memory Max Operating Temp         : 95 C
    Power Readings
        Power Management                  : Supported
        Power Draw                        : 68.37 W
        Power Limit                       : 400.00 W
        Default Power Limit               : 400.00 W
        Enforced Power Limit              : 400.00 W
        Min Power Limit                   : 100.00 W
        Max Power Limit                   : 400.00 W
    Clocks
        Graphics                          : 1155 MHz
        SM                                : 1155 MHz
        Memory                            : 1593 MHz
        Video                             : 1050 MHz
    Applications Clocks
        Graphics                          : 1155 MHz
        Memory                            : 1593 MHz
    Default Applications Clocks
        Graphics                          : 1155 MHz
        Memory                            : 1593 MHz
    Deferred Clocks
        Memory                            : N/A
    Max Clocks
        Graphics                          : 1410 MHz
        SM                                : 1410 MHz
        Memory                            : 1593 MHz
        Video                             : 1290 MHz
    Max Customer Boost Clocks
        Graphics                          : 1410 MHz
    Clock Policy
        Auto Boost                        : N/A
        Auto Boost Default                : N/A
    Voltage
        Graphics                          : 743.750 mV
    Fabric
        State                             : N/A
        Status                            : N/A
    Processes
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 280058
            Type                          : C
            Name                          : /gpfswork/rech/cnw/commun/conda/shared-m4-2023-02-28/bin/python
            Used GPU Memory               : 22374 MiB
```

don't see anything unusual

query all gpus for all their stats

```
nvidia-smi -q > gpu.query.all
```
then looked at clocks

the 0th gpu is at:
```
    Clocks
        Graphics                          : 1155 MHz
        SM                                : 1155 MHz
```
the other 7 are at:
```
    Clocks
        Graphics                          : 1410 MHz
        SM                                : 1410 MHz
```

perhaps it's normal since nothing happens to it as it's at 0% gpu util so it slowed itself down?

this one slightly off:
```
-        Memory Current Temp               : 45 C
+        Memory Current Temp               : 43 C
```
the 0th gpu at some point was at 45C, but that's far from very high, +80C is still fine, so it's not it.

query outputs:
- [gpu.query.0.txt](./gpu.query.0.txt)
- [gpu.query.1.txt](./gpu.query.1.txt)
- [gpu.query.all.txt](./gpu.query.all.txt)


Finally I wanted to see if anything was reported in the kernel logs, I run:
```
dmesg -H
```
but there was nothing from NVRM or nvidia around when it happened.

There are multiple failed kernel stacks for when I tried to attach to that sleeping process, but that's about it. See `dmesg` output:
- [dmesg.txt](./dmesg.txt)

The next day the sysadmins announced the filesystem had a meltdown, so this was the cause of node0:rank0 going into deep sleep - it tried to write and the system call never returned.
