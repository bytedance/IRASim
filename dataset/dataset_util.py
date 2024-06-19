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
import torch
import os
import math
import torch.distributed as dist
import numpy as np
import base64

from io import BytesIO
from PIL import Image

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def b64_2_img(data: str):
    image_b64 = base64.b64decode(data)
    img = Image.open(BytesIO(image_b64)).convert("RGB")
    return img


def get_continuous_action(d_acts, c_act_max, c_act_min, n_bins):
    c_act_max = c_act_max.to(d_acts.device)
    c_act_min = c_act_min.to(d_acts.device)
    c_acts = d_acts / (n_bins - 1) * (c_act_max - c_act_min) + c_act_min
    return c_acts




def alpha2rotm(a):
    """Alpha euler angle to rotation matrix."""
    rotm = np.array([
        [1, 0, 0],
        [0, np.cos(a), -np.sin(a)],
        [0, np.sin(a),  np.cos(a)]
    ])
    return rotm


def beta2rotm(b):
    """Beta euler angle to rotation matrix."""
    rotm = np.array([
        [np.cos(b), 0, np.sin(b)],
        [0, 1, 0],
        [-np.sin(b), 0, np.cos(b)]
    ])
    return rotm


def gamma2rotm(c):
    """Gamma euler angle to rotation matrix."""
    rotm = np.array([
        [np.cos(c), -np.sin(c), 0],
        [np.sin(c),  np.cos(c), 0],
        [0, 0, 1]
    ])
    return rotm


def euler2rotm(euler_angles):
    """Euler angle (ZYX) to rotation matrix."""
    alpha = euler_angles[0]
    beta = euler_angles[1]
    gamma = euler_angles[2]

    rotm_a = alpha2rotm(alpha)
    rotm_b = beta2rotm(beta)
    rotm_c = gamma2rotm(gamma)

    rotm = rotm_c @ rotm_b @ rotm_a

    return rotm


def isRotm(R):
    # Checks if a matrix is a valid rotation matrix.
    # Forked from Andy Zeng
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotm2euler(R):
    # Forked from: https://learnopencv.com/rotation-matrix-to-euler-angles/
    # R = Rz * Ry * Rx
    assert isRotm(R)
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
 
    if not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    
    # (-pi , pi]
    while x > np.pi:
        x -= (2 * np.pi)
    while x <= -np.pi:
        x += (2 * np.pi)
    while y > np.pi:
        y -= (2 * np.pi)
    while y <= -np.pi:
        y += (2 * np.pi)
    while z > np.pi:
        z -= (2 * np.pi)
    while z <= -np.pi:
        z += (2 * np.pi)
    return np.array([x, y, z])


def get_converted_fp32_paths(deepspeed_ckpt_path):
    deepspeed_ckpt_path = deepspeed_ckpt_path.rstrip('/')
    ckpt_dir = os.path.dirname(deepspeed_ckpt_path)
    ckpt_name = os.path.basename(deepspeed_ckpt_path)
    fp32_ckpt_name = f"{ckpt_name}.fp32.pt"
    converted_path = os.path.join(ckpt_dir, fp32_ckpt_name)
    return converted_path

def quat2rotm(quat):
    """Quaternion to rotation matrix.

    Args:
        quat (4, numpy array): quaternion x, y, z, w
    Returns:
        rotm (3x3 numpy array): rotation matrix
    """
    w = quat[3]
    x = quat[0]
    y = quat[1]
    z = quat[2]

    s = w*w + x*x + y*y + z*z

    rotm = np.array([[1-2*(y*y+z*z)/s, 2*(x*y-z*w)/s,   2*(x*z+y*w)/s  ],
                     [2*(x*y+z*w)/s,   1-2*(x*x+z*z)/s, 2*(y*z-x*w)/s  ],
                     [2*(x*z-y*w)/s,   2*(y*z+x*w)/s,   1-2*(x*x+y*y)/s]
    ])

    return rotm