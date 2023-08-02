# experiment 6.1

## TLDR

like 4.1, using old deeepspeed 0.6.7+patch, but also trying low grad clip of 0.3

Run into a node failure and slurm killing the training this time.

Crash-wise is inconclusive - thinking that this was an unrelated HW problem.

But we made a huge progress 2500 iterations and 8h of run! Definitely this is much much better!


### Setup

```
cd /gpfsdswork/projects/rech/cnw/commun/experiments/stas/m4-full
bash experiments/pretraining/vloom/tr_141_cm409xPMD01_scale_leap_of_faith_v5_num_workers_06/01_launch.sh

cd /gpfsssd/scratch/rech/cnw/commun/experiments/local_experiment_dir/tr_141_cm409xPMD01_scale_leap_of_faith_v5_num_workers_06/logs
tail -f main_log.txt
```


- m4@main - no code tweaks 4a5d063212acb6f255fa91bbf824ae5fbf89bbab / Tue Jan 17 22:46:56 2023 +0100
- conda env stas-m4
- install deepspeed==0.6.7 + patch
- pt-1.12
- grad_clip: 0.3
- CUDA_LAUNCH_BLOCKING=1
- num_workers=2
- normal accumulation (really using m4@main)

(also weirdly chose to do grad_clip: 2.0 - probably tried 1.0 and made a mistake - but it looked to make no negative impact on stability)


Using Deepspeed==v0.6.7 plus this [fix](https://github.com/microsoft/DeepSpeed/pull/2642)
```
commit 78a13fbf5b0ebc25b4d47c26c9ed8d9ac02d5eae (HEAD)
Author: Samyam Rajbhandari <samyamr@microsoft.com>
Date:   Thu Dec 22 16:50:45 2022 -0800

    [zero-3] Handle forward parameter return correctly in nested cases (#2642)

    Co-authored-by: Stas Bekman <stas00@users.noreply.github.com>
    Co-authored-by: Olatunji Ruwase <olruwase@microsoft.com>
    Co-authored-by: Jeff Rasley <jerasley@microsoft.com>
```

So specifically did:
```
source $cnw_ALL_CCFRWORK/start-m4-user
conda activate stas-m4
git clone https://github.com/microsoft/DeepSpeed DeepSpeed-v0.6.7
cd DeepSpeed-v0.6.7
pip install -e .
git checkout v0.6.7
git cherry-pick a298a43af22b9f971ff63e414887e659980889d9
```

### Investigation

a node crashed, so there was nothing we could do here - most likely a "normal" JZ hardware issue

```
iteration:  2500/500000   0% | elapsed time: 07:41:19 | per_token_loss: 3.1525 | lr: 9.990E-06 | num_tokens: 788088885 | num_images: 26284370 | num_padding: 391115 | fwd_bwd_time: 43621.8 | fwd_bwd_time_no_acc: 17.3 | image_to_text_ratio: 0.0332 | num_batches: 2500 | num_batches_in_curr_epoch: 706 | num_batches_since_training_logged: 25 | num_epochs: 1 | num_opt_steps: 2500 | z_loss: 24.6600 | per_example_loss: 15523.8 | pixel_values_sum: 1.54228E+12 | tflop_counter: 1.014E+08 | tflop_counter_no_acc: 4.055E+04 | tflops_fwd_bwd: 2.323E+03 | tflops_fwd_bwd_no_acc: 2.346E+03 | global_batch_size_current: 4096 |
** Starting validation **
Validation logs: val_per_token_loss: 3.0179 | val_per_example_loss: 14921.1 | val_num_images: 150355 | val_num_tokens: 4413196 | val_num_padding: 2292 | val_image_to_text_ratio: 0.0341 |
** Finished validation **
srun: error: Node failure on jean-zay-iam37
srun: Job step aborted: Waiting up to 62 seconds for job step to finish.
slurmstepd: error: *** STEP 1943824.0 ON jean-zay-iam17 CANCELLED AT 2023-03-06T03:17:51 DUE TO NODE FAILURE, SEE SLURMCTLD LOG FOR DETAILS ***
```
