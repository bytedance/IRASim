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

import os
import math
import torch
import logging
import random
import subprocess
import numpy as np
import torch.distributed as dist
import wandb
import argparse
# from torch._six import inf
from torch import inf
from PIL import Image
from typing import Union, Iterable
from collections import OrderedDict
from datetime import datetime
from omegaconf import OmegaConf

from diffusers.utils import is_bs4_available, is_ftfy_available

import html
import re
import urllib.parse as ul

if is_bs4_available():
    from bs4 import BeautifulSoup

if is_ftfy_available():
    import ftfy

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def optimizer_to_cpu(optimizer):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cpu()

def optimizer_to_gpu(optimizer, device='cuda:0'):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

def update_paths(args):
    args.base_dir = ''
    base_dir_list = ['']
    for base_dir in base_dir_list:
        if os.path.exists(base_dir):
            args.base_dir = base_dir

    args.dataset_dir = os.path.join(args.base_dir, args.dataset_dir)
    args.project_dir = os.path.join(args.base_dir, args.project_dir)

    args.train_annotation_path = os.path.join(args.dataset_dir, args.dataset, args.annotation_name,'train')
    args.test_annotation_path = os.path.join(args.dataset_dir, args.dataset, args.annotation_name, 'test')
    args.val_annotation_path = os.path.join(args.dataset_dir, args.dataset, args.annotation_name, 'val')
    args.video_path = os.path.join(args.dataset_dir, args.dataset)

    args.results_dir = os.path.join(args.project_dir, args.results_dir)
    args.vae_model_path = os.path.join(args.project_dir, args.vae_model_path)

    if args.evaluate_checkpoint:
        args.evaluate_checkpoint = os.path.join(args.dataset_dir, args.evaluate_checkpoint)
    if args.resume_from_checkpoint:
        args.resume_from_checkpoint = os.path.join(args.results_dir, args.resume_from_checkpoint)

    # evaluation
    args.fid_model_path = os.path.join(args.dataset_dir,'evaluation_model', 'pt_inception-2015-12-05-6726825d.pth')
    args.fvd_model_path = os.path.join(args.dataset_dir,'evaluation_model', 'i3d_torchscript.pt')
    args.fid_cache_path = os.path.join(args.dataset_dir,args.dataset,'evaluation_cache', f"{args.mode}_fid_cache.npz")

    # true samples path
    args.true_sample_latent_videos_dir = f'{args.dataset_dir}/{args.dataset}/evaluation_latent_videos/{args.mode}_sample_latent_videos'
    args.true_sample_videos_dir = f'{args.dataset_dir}/{args.dataset}/evaluation_videos/{args.mode}_sample_videos'
    args.true_sample_frames_dir = f'{args.dataset_dir}/{args.dataset}/evaluation_videos/{args.mode}_sample_frames'

    # true episodes path
    args.true_episode_latent_videos_dir = f'{args.dataset_dir}/{args.dataset}/latent_videos/{args.mode}'
    args.true_episode_videos_dir = f'{args.dataset_dir}/{args.dataset}/videos/{args.mode}'

    

def get_args(args):
    data_config = OmegaConf.load("configs/base/data.yaml")
    diffusion_config = OmegaConf.load("configs/base/diffusion.yaml")
    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(data_config, config)
    config = OmegaConf.merge(diffusion_config, config)
    args = config
    update_paths(args)
    return args

#################################################################################
#                             Training Clip Gradients                           #
#################################################################################

def get_grad_norm(
        parameters: _tensor_or_tensors, norm_type: float = 2.0) -> torch.Tensor:
    r"""
    Copy from torch.nn.utils.clip_grad_norm_

    Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    device = grads[0].device
    if norm_type == inf:
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
    return total_norm

def clip_grad_norm_(
        parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False, clip_grad = True) -> torch.Tensor:
    r"""
    Copy from torch.nn.utils.clip_grad_norm_

    Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    device = grads[0].device
    if norm_type == inf:
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)

    if clip_grad:
        if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
            raise RuntimeError(
                f'The total norm of order {norm_type} for gradients from '
                '`parameters` is non-finite, so it cannot be clipped. To disable '
                'this error and scale the gradients by the non-finite norm anyway, '
                'set `error_if_nonfinite=False`')
        clip_coef = max_norm / (total_norm + 1e-6)
        # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
        # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
        # when the gradients do not reside in CPU memory.
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for g in grads:
            g.detach().mul_(clip_coef_clamped.to(g.device))
        # gradient_cliped = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
        # print(gradient_cliped)
    return total_norm

def setup_experiment_dir(rank, args):
    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    current_date = datetime.now()
    experiment_dir = f"{args.results_dir}/{current_date.strftime('%m')}/{current_date.strftime('%d')}/{args.anno}"
    # experiment_dir = f"{args.results_dir}/06/11/{args.anno}"
    if 'debug' not in args.anno or not args.debug:
        exp_num = 0
        os.makedirs(os.path.dirname(experiment_dir),exist_ok=True)
        for i in os.listdir(os.path.dirname(experiment_dir)):
            if i.startswith(args.anno):
                if len(i) > len(args.anno) and i[len(args.anno)] != '0':
                    continue
                else:
                    exp_num += 1
        if exp_num !=0:
            experiment_dir = experiment_dir+'{:03d}'.format(exp_num)

    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    videos_dir = f"{experiment_dir}/videos"  # Stores saved model checkpoints
    # Setup an experiment folder:
    wandb_name = '_'.join(experiment_dir.split('/')[-3:])
    if rank == 0 or args.debug:
        wandb.login(key='') # TODO setup your own wandb key 
        wandb.init(project=args.dataset, entity="", name = wandb_name)
        if 'debug' in checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(videos_dir, exist_ok=True)
        else:
            if os.path.exists(checkpoint_dir):
                assert False
            os.makedirs(checkpoint_dir, exist_ok=False)
            os.makedirs(videos_dir, exist_ok=False)
            pass
        logger = create_logger(experiment_dir,args)
        # writer = create_wandb(experiment_dir)    
        OmegaConf.save(args, os.path.join(experiment_dir, 'config.yaml'))
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None,args)
        # writer = None

    return logger, checkpoint_dir, videos_dir, wandb_name

#################################################################################
#                             Training Logger                                   #
#################################################################################

def create_logger(logging_dir,args):
    """
    Create a logger that writes to a log file and stdout.
    """
    if args.do_evaluate:
        train_steps = int(args.evaluate_checkpoint.split('/')[-1][0:-3])
        log_name = f'log_{args.mode}_{train_steps}.txt'
    else:
        log_name = f'log.txt'
    if dist.get_rank() == 0 and os.environ["WORLD_SIZE"] != '1':  # real logger
        logging.basicConfig(
            level=logging.INFO,
            # format='[\033[34m%(asctime)s\033[0m] %(message)s',
            format='[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/{log_name}")]
        )
        logger = logging.getLogger(__name__)
        
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger




#################################################################################
#                      EMA Update/ DDP Training Utils                           #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()
    

def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            # os.environ["MASTER_PORT"] = "29566"
            os.environ["MASTER_PORT"] = str(29567 + num_gpus)
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        try:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
        except:
            print("Can not find RANK and WORLD_SIZE, Debug Mode")
            os.environ["CUDA_VISIBLE_DEVICES"] = '1'
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "9003"
            os.environ["LOCAL_RANK"] = "0"
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])


    # torch.cuda.set_device(rank % num_gpus)
    
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )






    


