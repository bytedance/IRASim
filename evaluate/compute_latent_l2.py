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

import cv2
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent
import torch
import torch.nn.functional as F

def process_video(file, pred_folder, true_folder):
    pred_latent_path = os.path.join(pred_folder, file)
    assert os.path.exists(pred_latent_path)

    true_latent_path = os.path.join(true_folder, file)
    if not os.path.exists(true_latent_path):
        true_latent_path = os.path.join(true_folder, file.split('.')[0],'0.pt')
    
    assert os.path.exists(true_latent_path)


    with open(pred_latent_path,'rb') as file:
        pred_latent = torch.load(file)

    if 'languagetable' in true_folder and 'sample' not in true_folder:
        with open(true_latent_path,'rb') as file:
            true_latent = torch.load(file)['obs']
    else:
        with open(true_latent_path,'rb') as file:
            true_latent = torch.load(file)
    vae_l2 = F.mse_loss(pred_latent[1:], true_latent[1:]).item()
    return vae_l2


def compute_latent_l2(pred_folder, true_folder):
    files1 = os.listdir(pred_folder)
    overall_l2 = []
    results = {}
    with ThreadPoolExecutor(max_workers=64) as executor:
        future_to_file = {executor.submit(process_video, file, pred_folder, true_folder): file for file in files1}
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(files1)):
            file = future_to_file[future]
            result = future.result()
            overall_l2.append(result)

    total_l2 = sum(overall_l2) / len(overall_l2) if overall_l2 else 0

    return total_l2

# total_l2= compute_latent_l2(pred_folder, true_folder)
# print(pred_folder)
# print(f"Overall Average Latent L2: {total_l2:.4f}")
