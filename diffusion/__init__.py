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

# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from . import mask_gaussian_diffusion as mask_gd
from .mask_respace import MaskSpacedDiffusion, space_timesteps


def create_mask_diffusion(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    # learn_sigma=False,
    rescale_learned_sigmas=False,
    diffusion_steps=1000
):
    betas = mask_gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = mask_gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = mask_gd.LossType.RESCALED_MSE
    else:
        loss_type = mask_gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return MaskSpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            mask_gd.ModelMeanType.EPSILON if not predict_xstart else mask_gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                mask_gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else mask_gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else mask_gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        # rescale_timesteps=rescale_timesteps,
    )