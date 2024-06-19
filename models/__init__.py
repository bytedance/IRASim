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
from baselines.vdm.video_diffusion_pytorch import Unet3D_Trajectory
from models.irasim import IRASim_models
from torch.optim.lr_scheduler import LambdaLR


def customized_lr_scheduler(optimizer, warmup_steps=5000): # 5000 from u-vit
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'warmup':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)
    
def get_models(args):
    if 'IRASim' in args.model:
        return IRASim_models[args.model](
                input_size=args.latent_size,
                num_frames=args.num_frames,
                learn_sigma=args.learn_sigma,
                extras=args.extras,
                attention_mode = args.attention_mode,
                args = args
            )
    elif 'LVDM' in args.model and args.pre_encode:
        return Unet3D_Trajectory(
                dim = args.lvdm_dim, # 288
                dim_mults = (1, 2, 4, 8),
                channels = 4,
                cond_dim = 768,
                args = args
            )
    elif 'VDM' in args.model and not args.pre_encode:
        return Unet3D_Trajectory(
                dim = 64,
                dim_mults = (1, 2, 4, 8),
                channels = 3,
                cond_dim = 768,
                args = args
            )
    else:
        raise '{} Model Not Supported!'.format(args.model)
    