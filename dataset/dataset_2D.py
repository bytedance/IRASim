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

import pickle
import os
import argparse
from omegaconf import OmegaConf
import random
import warnings
import traceback
from tqdm import tqdm
import torch
import numpy as np
import torch
import pickle
import json
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as T

from decord import VideoReader, cpu
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataset.video_transforms import Resize_Preprocess, ToTensorVideo
from util import update_paths

class Dataset_2D(Dataset):
    def __init__(self,args,mode):
        """Constructor."""
        super().__init__()
        self.args = args
        if mode == 'train':
            self.dataset_dir = args.train_annotation_path
            self.start_frame_interval = 1
        elif mode == 'val':
            self.dataset_dir = args.val_annotation_path
            self.start_frame_interval = args.val_start_frame_interval
        elif mode == 'test':
            self.dataset_dir = args.test_annotation_path
            self.start_frame_interval = args.val_start_frame_interval
        self.video_path = args.video_path
        self.sequence_interval = args.sequence_interval
        assert self.sequence_interval == 1
        self.seq_len = args.num_frames-1
        self.c_act_scaler = [20.0, 20.0]
        self.c_act_scaler = np.array(self.c_act_scaler, dtype=float)
        # self.crop = crop
        self.action_dim = 2  # ee xyz (3) + ee euler (3) + gripper(1)

        self.ann_files = self._init_anns(self.dataset_dir)
        self.ann_files = sorted(self.ann_files, key=lambda x: x.split('/')[-1])

        self.samples = self._init_samples(self.ann_files)
        self.samples = sorted(self.samples, key=lambda x: (x['ann_file'], x['frame_ids'][0]))


        if self.args.debug and not self.args.do_evaluate:
            self.samples = self.samples[0:10]
        print(f'{len(self.ann_files)} trajectories in total')
        print(f'{len(self.samples)} samples in total')
        self.error_num = 0
        self.training = False
        self.preprocess = T.Compose([
            ToTensorVideo(),
            Resize_Preprocess(tuple(args.video_size)), # 288 512
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.resize_preprocess = T.Compose([
            Resize_Preprocess(tuple(args.video_size))
        ])

    def __str__(self):
        return f"{len(self.ann_files)} samples from {self.data_path}"

    def _init_anns(self, data_dir):
        files = []
        for root, dirs, files_in_dir in os.walk(data_dir):
            for file in files_in_dir:
                full_path = os.path.join(root, file)
                files.append(full_path)
        return files


    def _load_and_process_ann_file(self, ann_file):
        samples = []
        try:
            with open(ann_file, "rb") as f:
                ann = json.load(f)
        except:
            print(ann_file)
            assert False, 'wrong file'

        n_frames = len(ann['actions'])
        for frame_i in range(0, n_frames, self.start_frame_interval):
            sample = dict()
            sample['ann_file'] = ann_file
            sample['frame_ids'] = []
            curr_frame_i = frame_i
            while True:
                if curr_frame_i > (n_frames - 1):
                    break
                sample['frame_ids'].append(curr_frame_i)
                if len(sample['frame_ids']) == self.args.num_frames:
                    break
                curr_frame_i += self.sequence_interval
            # make sure there are sequence_length number of frames
            if len(sample['frame_ids']) == self.args.num_frames:
                samples.append(sample)
        return samples

    # def _load_and_process_ann_file(self, ann_file):
    #     samples = []
    #     try:
    #         with open(ann_file, "rb") as f:
    #             ann = torch.load(f)
    #     except:
    #         print(ann_file)
    #         assert False, 'wrong file'

    #     n_frames = len(ann['actions'])
    #     if n_frames < self.args.num_frames:
    #         return []
    #     for frame_i in range(0, n_frames-self.args.num_frames+1, self.start_frame_interval):
    #         sample = dict()
    #         sample['ann_file'] = ann_file
    #         sample['frame_start_id'] = frame_i
    #         samples.append(sample)
    #     return samples

    def _init_samples(self, ann_files):
        samples = []
        with ThreadPoolExecutor(32) as executor:
            future_to_ann_file = {executor.submit(self._load_and_process_ann_file, ann_file): ann_file for ann_file in ann_files}
            for future in tqdm(as_completed(future_to_ann_file), total=len(ann_files), desc='load dataset'):
                samples.extend(future.result())
        return samples

    def __len__(self):
        return len(self.samples)

    def get_latent(self,ann,frame_ids):
        latent_video_path = os.path.join(self.video_path, ann['latent_video_path'])
        with open(latent_video_path,'rb') as file:
            frames = torch.load(file)['obs']
        frames = frames[frame_ids]
        return frames
    
    def get_video(self,ann,frame_ids):
        video_path = os.path.join(self.video_path,ann['video_path'])
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        assert (np.array(frame_ids) < len(vr)).all()
        assert (np.array(frame_ids) >= 0).all()
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        frames = torch.from_numpy(frame_data)
        frames = frames.permute(0, 3, 1, 2)
        if self.args.normalize:
            frames = self.preprocess(frames)
        else:
            frames = self.resize_preprocess(frames)
        return frames

    def __getitem__(self, index, cam_id = None, return_video = False):
        # Make sure validation data are the same
        if not self.training:
            np.random.seed(index)
            random.seed(index)
        try:
            sample = self.samples[index]
            ann_file = sample['ann_file']
            
            frame_ids = sample['frame_ids']
            with open(ann_file, "rb") as f:
                ann = json.load(f)
            episode_id = ann['episode_id']
            actions = np.array(ann['actions'])[frame_ids[0:-1]]
            actions *= self.c_act_scaler


            assert actions.shape[0] == self.seq_len, f'actions shape[0] = {actions.shape[0]}'

            data = dict()
            data['action'] = torch.from_numpy(actions).float()
            if self.args.pre_encode:
                latent = self.get_latent(ann,frame_ids)
                data['latent'] = latent.float()
                if return_video:
                    video = self.get_video(ann,frame_ids)
                    data['video'] = video.float()
            else:
                video = self.get_video(ann,frame_ids)
                data['video'] = video.float()

            data['video_name'] = {'episode_id': episode_id, 'start_frame_id': str(frame_ids[0]),'cam_id':'0'}
            return data

        except Exception:
            warnings.warn(f"Invalid data encountered: {self.samples[index]}. Skipped "
                          f"(by randomly sampling another sample in the same dataset).")
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            self.error_num += 1
            print(self.error_num)
            return self[np.random.randint(len(self.samples))]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/evaluation/languagetable/frame_ada.yaml")
    args = parser.parse_args()
    data_config = OmegaConf.load("configs/base/data.yaml")
    diffusion_config = OmegaConf.load("configs/base/diffusion.yaml")
    args = OmegaConf.load(args.config)
    args = OmegaConf.merge(data_config, args)
    args = OmegaConf.merge(diffusion_config, args)
    update_paths(args)

    dataset = Dataset_2D(args,mode='test')

    data_loader = DataLoader(dataset=dataset, 
                                    batch_size=1, 
                                    shuffle=True, 
                                    num_workers=0)
    video_name_set = set()
    for data in tqdm(data_loader,total=len(data_loader)):
        print(data['latent'].size())
        