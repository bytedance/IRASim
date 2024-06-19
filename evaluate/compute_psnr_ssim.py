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
import torch
import numpy as np
import concurrent
import torchvision.transforms.functional as F
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio
from concurrent.futures import ThreadPoolExecutor


# def calculate_psnr(img1, img2):
#     if img1.shape != img2.shape:
#         img2_resized = F.resize(torch.from_numpy(img2/255.0), img1.shape, antialias=True)
#         img2 = np.array(img2_resized)
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return float('inf')
#     max_pixel = 255.0
#     return 20 * np.log10(max_pixel / np.sqrt(mse))


def calculate_psnr(img1, img2):
    if img1.shape != img2.shape:
        img2_resized = F.resize(torch.from_numpy(img2).permute(2,0,1), img1.shape[0:2], antialias=True)
        img2 = np.array(img2_resized.permute(1,2,0).to(torch.uint8)) # 将其放回到0-255的范围内
    psnr = peak_signal_noise_ratio(img1, img2, data_range=255)
    return psnr


def process_video_psnr(file, folder1, folder2):
    video_path1 = os.path.join(folder1, file)
    video_path2 = os.path.join(folder2, file)
    if not os.path.exists(video_path2):
        video_path2 = os.path.join(folder2, file[0:-4],'rgb.mp4')
    assert os.path.exists(video_path1)
    assert os.path.exists(video_path2)
        

    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    psnr_scores = []
    
    idx = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if idx == 0:
            idx +=1
            continue
        if not ret1 or not ret2:
            break

        psnr_score = calculate_psnr(frame1, frame2)

        psnr_scores.append(psnr_score)
    
    cap1.release()
    cap2.release()
    
    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    
    return file, {'PSNR': avg_psnr}

def process_video_psnr_ssim(file, folder1, folder2):
    video_path1 = os.path.join(folder1, file)
    video_path2 = os.path.join(folder2, file)

    if not os.path.exists(video_path2):
        assert False

    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    psnr_scores = []
    ssim_scores = []
    idx = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if idx == 0:
            idx +=1
            continue
        if not ret1 or not ret2:
            break

        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        psnr_score = calculate_psnr(frame1_gray, frame2_gray)
        ssim_score = ssim(frame1_gray, frame2_gray)

        psnr_scores.append(psnr_score)
        ssim_scores.append(ssim_score)
    
    cap1.release()
    cap2.release()
    
    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    avg_ssim = sum(ssim_scores) / len(ssim_scores)
    
    return file, {'PSNR': avg_psnr, 'SSIM': avg_ssim}


def compute_psnr(folder1, folder2):
    files1 = {file for file in os.listdir(folder1) if file.endswith('.mp4')}
    files2 = set(os.listdir(folder2))
    print(f'pred long video number: {len(files1)}')
    print(f'true long video number: {len(files2)}')
    overall_psnr = []

    results = {}
    with ThreadPoolExecutor(max_workers=64) as executor:
        future_to_file = {executor.submit(process_video_psnr, file, folder1, folder2): file for file in files1}
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(files1)):
            file = future_to_file[future]
            result = future.result()
            results[result[0]] = result[1]
            overall_psnr.append(result[1]['PSNR'])

    total_avg_psnr = sum(overall_psnr) / len(overall_psnr) if overall_psnr else 0

    return total_avg_psnr

def compute_psnr_ssim(folder1, folder2):
    files1 = {file for file in os.listdir(folder1) if file.endswith('.mp4')}
    files2 = set(os.listdir(folder2))
    print(f'pred long video number: {len(files1)}')
    print(f'true long video number: {len(files2)}')

    overall_psnr = []
    overall_ssim = []

    results = {}
    with ThreadPoolExecutor(max_workers=64) as executor:
        future_to_file = {executor.submit(process_video_psnr_ssim, file, folder1, folder2): file for file in files1}
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(files1)):
            file = future_to_file[future]
            result = future.result()
            results[result[0]] = result[1]
            overall_psnr.append(result[1]['PSNR'])
            overall_ssim.append(result[1]['SSIM'])

    total_avg_psnr = sum(overall_psnr) / len(overall_psnr) if overall_psnr else 0
    total_avg_ssim = sum(overall_ssim) / len(overall_ssim) if overall_ssim else 0

    return total_avg_psnr, total_avg_ssim


