# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A minimal training script for IRASim using PyTorch DDP.
"""

import os
import math
import argparse
import torch
import imageio
import wandb
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from time import time
from copy import deepcopy
from einops import rearrange
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.schedulers import PNDMScheduler
from models import get_models
from dataset import get_dataset
from diffusion import create_mask_diffusion
from util import (
    optimizer_to_cpu, optimizer_to_gpu, clip_grad_norm_,
    update_ema, requires_grad, cleanup, setup_distributed,
    setup_experiment_dir, get_args
)
from sample.pipeline_trajectory2videogen import Trajectory2VideoGenPipeline
from evaluate.generate_short_video import generate_sample_videos

# Maybe use fp16 precision training need to set to False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
#################################################################################
#                                  Training Loop                                #
#################################################################################


def validation(args, val_dataloader, device, vae, diffusion, model):
    total_loss = []
    for video_data in tqdm(val_dataloader,total=len(val_dataloader)):
        mask_frame_num = args.mask_frame_num
        if not args.pre_encode:
            x = video_data['video'].to(device, non_blocking=True)
            with torch.no_grad():
                b, f, _, _, _ = x.shape
                x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
                stride = args.local_val_batch_size
                encode_video_list = []
                for i in range(0,x.size()[0],stride):
                    encode_video_list.append(vae.encode(x[i:i+stride]).latent_dist.sample().mul_(vae.config.scaling_factor))
                encode_video = torch.cat(encode_video_list,dim=0)
                x = rearrange(encode_video, '(b f) c h w -> b f c h w',b=b,f=f)
        else:
            x = video_data['latent'].to(device, non_blocking=True)

        if args.extras == 3:
            actions = video_data['action'].to(device, non_blocking=True)
            model_kwargs = dict(actions=actions,mask_frame_num = mask_frame_num)
        elif args.extras == 5:
            actions = video_data['action'].to(device, non_blocking=True)
            model_kwargs = dict(actions=actions, mask_frame_num = mask_frame_num) 

        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
        loss = loss_dict["loss"].mean()
    total_loss.append(loss)
    return sum(total_loss)/len(total_loss)


def validate_video_generation(val_dataset, args, vae, model, device, train_steps, videos_dir, wandb_name):
    batch_id = list(range(0,len(val_dataset),int(len(val_dataset)/3)))
    batch_list = [val_dataset.__getitem__(id, cam_id=0, return_video = True) for id in batch_id]
    actions = torch.cat([t['action'].unsqueeze(0) for i, t in enumerate(batch_list) ],dim=0).to(device, non_blocking=True)
    true_video = torch.cat([t['video'].unsqueeze(0) for i,t in enumerate(batch_list)],dim=0).to(device, non_blocking=True)
    

    mask_frame_num = args.mask_frame_num
    if not args.pre_encode:
        mask_x = true_video[:,0:mask_frame_num]
        with torch.no_grad():
            b, f, _, _, _ = mask_x.shape
            mask_x = rearrange(mask_x, 'b f c h w -> (b f) c h w').contiguous()
            mask_x = vae.encode(mask_x).latent_dist.sample().mul_(vae.config.scaling_factor)
            mask_x = rearrange(mask_x, '(b f) c h w -> b f c h w',b=b,f=f)
    else:
        latent = torch.cat([t['latent'].unsqueeze(0) for i,t in enumerate(batch_list)],dim=0).to(device, non_blocking=True)
        mask_x = latent[:,0:mask_frame_num]

    #################
    # denoise x
    #################

    if args.sample_method == 'PNDM':
        scheduler = PNDMScheduler.from_pretrained(args.scheduler_path, beta_start=args.beta_start, beta_end=args.beta_end, beta_schedule=args.beta_schedule, variance_type=args.variance_type)
    elif args.sample_method == 'DDPM':
        scheduler = PNDMScheduler.from_pretrained(args.scheduler_path, beta_start=args.beta_start, beta_end=args.beta_end, beta_schedule=args.beta_schedule, variance_type=args.variance_type)

    videogen_pipeline = Trajectory2VideoGenPipeline(vae=vae, scheduler=scheduler, transformer=model)

    print('Generation total {} videos'.format(mask_x.size()[0]))
    videos, latents = videogen_pipeline(
                            actions, 
                            mask_x = mask_x,
                            video_length=args.num_frames, 
                            height=args.video_size[0], 
                            width=args.video_size[1], 
                            num_inference_steps=args.infer_num_sampling_steps,
                            guidance_scale=args.guidance_scale, # dumpy
                            device = device,
                            output_type = 'both'
                            )

    
    def printvideo(videos,filename):
        t_videos = rearrange(videos, 'b f c h w -> b f h w c')
        t_videos = ((t_videos[0] / 2.0 + 0.5).clamp(0, 1) * 255).detach().to(dtype=torch.uint8).cpu().contiguous().numpy()
        print(t_videos.shape)
        writer = imageio.get_writer(filename, fps=4) 
        for frame in t_videos:
            writer.append_data(frame)

    videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255).detach().to(dtype=torch.uint8).cpu().contiguous()
    true_video = ((true_video / 2.0 + 0.5).clamp(0, 1) * 255).detach().to(dtype=torch.uint8).cpu().contiguous()
    videos = torch.cat([true_video[:,0:mask_frame_num],videos[:,mask_frame_num:]],dim=1)
    videos = torch.cat([true_video, videos],dim=-2)
    videos = rearrange(videos, 'b f c h w -> b f h w c')
    cated_videos = torch.cat(torch.split(videos,split_size_or_sections=1,dim=0),dim=-2).squeeze().numpy()
    filename = f"{videos_dir}/train_steps_{train_steps}.mp4"
    writer = imageio.get_writer(filename, fps=4) # fps 是帧率
    for frame in cated_videos:
        writer.append_data(frame)
    writer.close()
    wandb.log({f"{wandb_name}_train_steps_{train_steps}": wandb.Video(filename, fps=4, format="mp4")})
    return 

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    if args.debug or args.do_evaluate:
        args.anno = args.anno + '-debug'
        os.environ["WANDB_MODE"] = "offline"
        args.log_every = 20
        args.val_every = 20

    # Setup DDP:
    setup_distributed()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    seed = args.global_seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, local rank={local_rank}, seed={seed}, world_size={dist.get_world_size()}.")


    logger, checkpoint_dir, videos_dir, wandb_name = setup_experiment_dir(rank,args)

    # Create model:
    args.latent_size = [t // 8 for t in args.video_size]
    model = get_models(args)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_mask_diffusion(timestep_respacing="",learn_sigma=args.learn_sigma)
    vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device)


    assert not (args.evaluate_checkpoint == False and args.do_evaluate)
    if args.evaluate_checkpoint:
        checkpoint = torch.load(args.evaluate_checkpoint, map_location=lambda storage, loc: storage)
        train_steps = int(args.evaluate_checkpoint.split('/')[-1][0:-3])
        if "ema" in checkpoint:  # supports checkpoints from train.py
            logger.info('Using ema ckpt!')
            checkpoint = checkpoint["ema"]

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict:
                pretrained_dict[k] = v
            else:
                logger.info('Ignoring: {}'.format(k))
        logger.info('Successfully Load {}% original pretrained model weights '.format(len(pretrained_dict) / len(checkpoint.items()) * 100))
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info('Successfully load model at {}!'.format(args.evaluate_checkpoint))


    model = DDP(model.to(device), device_ids=[local_rank])

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0)

    vae.requires_grad_(False)

    # Setup data:
    train_dataset,val_dataset = get_dataset(args)
    if train_dataset!=None:
        train_dataset.training = True
    if train_dataset is not None:
        train_sampler = DistributedSampler(train_dataset,num_replicas=dist.get_world_size(),rank=rank,shuffle=True,seed=args.global_seed)
        train_dataloader = DataLoader(train_dataset,batch_size=int(args.local_batch_size),shuffle=False,sampler=train_sampler,
            num_workers=args.num_workers,pin_memory=True,drop_last=True)
    val_sampler = DistributedSampler(val_dataset,num_replicas=dist.get_world_size(),rank=rank,shuffle=False,seed=args.global_seed)
    val_dataloader = DataLoader(val_dataset,batch_size=int(args.local_val_batch_size),shuffle=False,sampler=val_sampler,
        num_workers=args.num_workers,pin_memory=True,drop_last=False)

    # logger.info(f"Dataset contains {len(train_dataset):,} videos ({args.train_annotation_path})")
    # logger.info(f"Dataset contains {len(val_dataset):,} videos ({args.val_annotation_path})")
    if train_dataset is not None:
        logger.info(f'{len(train_dataset.ann_files)} trajectories in Train')
        logger.info(f'{len(train_dataset.samples)} samples in Train')
    logger.info(f'{len(val_dataset.ann_files)} trajectories in Valdation')
    logger.info(f'{len(val_dataset.samples)} samples in Valdation')


    # Scheduler
    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=opt,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare models for training:
    # TODO
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    if args.do_evaluate:
        model.cpu()
        optimizer_to_cpu(opt)
        torch.cuda.empty_cache()
        pred_video_base_dir = f"{checkpoint_dir}/{train_steps:07d}/{args.mode}_sample_videos"
        pred_latent_video_base_dir = f"{checkpoint_dir}/{train_steps:07d}/{args.mode}_sample_latents"
        pred_frame_base_dir = f"{checkpoint_dir}/{train_steps:07d}/{args.mode}_sample_frames"
        logger.info('generating sample videos')
        generate_sample_videos(args, val_dataloader, device, vae, ema, pred_video_base_dir,pred_frame_base_dir,pred_latent_video_base_dir)
        dist.barrier()
        torch.cuda.empty_cache()
        cleanup()
        return


    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    first_epoch = 0
    start_time = time()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = (math.ceil(len(train_dataloader)) / args.gradient_accumulation_steps)# math.ceil(2314893/32) *32
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        checkpoint_path = os.path.join(args.resume_from_checkpoint, 'checkpoints')
        dirs = os.listdir(checkpoint_path)
        dirs = [d for d in dirs if d.endswith("pt")]
        dirs = sorted(dirs, key=lambda x: int(x.split(".")[0]))
        path = dirs[-1]
        checkpoint_path = os.path.join(checkpoint_path,path) 
        logger.info(f"Resuming from checkpoint {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            logger.info('Using ema ckpt!')
            # checkpoint = checkpoint["ema"]
        model.module.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])
        train_steps = int(path.split(".")[0])
        first_epoch = int(train_steps // num_update_steps_per_epoch)
        resume_step = (train_steps % num_update_steps_per_epoch)*args.gradient_accumulation_steps
        # resume_sample_num = (train_steps * 32) % 2314893

    if args.evaluate_checkpoint:
        train_steps = int(args.evaluate_checkpoint.split("/")[-1].split('.')[0])
    accumulation_steps = args.gradient_accumulation_steps
    accumulated_steps = 0  
    for epoch in range(first_epoch, num_train_epochs):
        train_sampler.set_epoch(epoch)
        for step, video_data in enumerate(train_dataloader):
            mask_frame_num = args.mask_frame_num
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                continue

            video_name = video_data['video_name']
            if not args.pre_encode:
                x = video_data['video'].to(device, non_blocking=True)
                with torch.no_grad():
                    b, _, _, _, _ = x.shape
                    x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
                    x = vae.encode(x).latent_dist.sample().mul_(vae.config.scaling_factor)
                    x = rearrange(x, '(b f) c h w -> b f c h w', b=b).contiguous()
            else:
                x = video_data['latent'].to(device, non_blocking=True)

            if args.extras == 3:
                actions = video_data['action'].to(device, non_blocking=True)
                if args.dataset == 'droid':
                    actions = torch.cat([actions,actions],dim=0)
                model_kwargs = dict(actions=actions,mask_frame_num = mask_frame_num)
            elif args.extras == 5:
                actions = video_data['action'].to(device, non_blocking=True)
                model_kwargs = dict(actions=actions, mask_frame_num = mask_frame_num) 

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean() / accumulation_steps
            
            loss.backward()

            running_loss += loss.item() 
            accumulated_steps += 1

            if accumulated_steps % accumulation_steps == 0:
                if train_steps < args.start_clip_iter:
                    gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=False)
                else:
                    gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=True)
                

                opt.step()
                lr_scheduler.step()
                opt.zero_grad()
                update_ema(ema, model.module)
                accumulated_steps = 0 
                train_steps += 1
                log_steps += 1

                if train_steps % args.log_every == 0:
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    
                    avg_loss = avg_loss.item() / dist.get_world_size()
                    


                    logger.info(f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_loss:.4f}, Gradient Norm: {gradient_norm:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    if rank == 0:
                        wandb.log({'Train Loss': avg_loss}, train_steps)
                        wandb.log({'Gradient Norm': gradient_norm}, train_steps)
                    # Reset monitoring variables:
                    running_loss = 0
                    log_steps = 0
                    start_time = time()
                
                if train_steps % args.val_every == 0:
                    
                    if not (args.debug and args.do_evaluate):
                        avg_loss = validation(args, val_dataloader, device, vae, diffusion, ema)
                        avg_loss = avg_loss.clone().detach()
                        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                        avg_loss = avg_loss.item() / dist.get_world_size()
                        if rank == 0:
                            wandb.log({'All Reduce Val Loss': avg_loss}, train_steps)
                            validate_video_generation(val_dataset, args, vae, ema, device, train_steps, videos_dir, wandb_name)
                    dist.barrier()

                # Save IRASim checkpoint:
                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    if rank == 0:
                        # model.cpu()
                        # ema.cpu()
                        # optimizer_to_cpu(opt)
                        checkpoint = {
                            "model": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "args": args
                        }
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        # model.to(device)
                        # ema.to(device)
                        # optimizer_to_gpu(opt,device)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    dist.barrier()
            

    model.eval()

    logger.info("Done!")
    cleanup()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/bridge/frame_ada.yaml")
    args = parser.parse_args()
    args = get_args(args)
    main(args)

