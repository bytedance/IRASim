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

import json
import os
import random
import warnings
import traceback
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms as T
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import imageio
from decord import VideoReader, cpu
from dataset.dataset_util import euler2rotm, rotm2euler
from concurrent.futures import ThreadPoolExecutor, as_completed
from einops import rearrange
from dataset.video_transforms import Resize_Preprocess, ToTensorVideo
from util import update_paths


class Dataset_3D(Dataset):
    def __init__(
            self,
            args,mode = 'val'
    ):
        """Constructor."""
        super().__init__()
        self.args = args
        if mode == 'train':
            self.data_path = args.train_annotation_path
            self.start_frame_interval = 1
        elif mode == 'val':
            self.data_path = args.val_annotation_path
            self.start_frame_interval = args.val_start_frame_interval
        elif mode == 'test':
            self.data_path = args.test_annotation_path
            self.start_frame_interval = args.val_start_frame_interval
        self.video_path = args.video_path
        self.sequence_interval = args.sequence_interval
        self.mode = mode
        self.sequence_length = args.num_frames
        
        self.cam_ids = args.cam_ids
        self.accumulate_action = args.accumulate_action

        self.action_dim = 7  # ee xyz (3) + ee euler (3) + gripper(1)
        self.c_act_scaler = [20.0, 20.0,20.0, 20.0,20.0, 20.0, 1.0]
        self.c_act_scaler = np.array(self.c_act_scaler, dtype=float)
        self.ann_files = self._init_anns(self.data_path)

        # total_action_num = 0
        # total_ann_num = len(self.ann_files)
        # for ann_file in tqdm(self.ann_files,total = total_ann_num):
        #     with open(ann_file, "rb") as f:
        #         ann = json.load(f)
        #     antion_num = len(ann['action'])
        #     if antion_num < 15:
        #         total_ann_num -= 1
        #     else:
        #         total_action_num += antion_num
        # avg_action_num = total_action_num/total_ann_num


        print(f'{len(self.ann_files)} trajectories in total')
        self.samples = self._init_sequences(self.ann_files)
        
        self.samples = sorted(self.samples, key=lambda x: (x['ann_file'], x['frame_ids'][0]))
        if args.debug and not args.do_evaluate:
            self.samples = self.samples[0:10]
        print(f'{len(self.ann_files)} trajectories in total')
        print(f'{len(self.samples)} samples in total')
        # with open('./samples_16.pkl','wb') as file:
        #     pickle.dump(self.samples,file)
        self.wrong_number = 0
        self.transform = T.Compose([
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.training = False
        self.preprocess = T.Compose([
            ToTensorVideo(),
            Resize_Preprocess(tuple(args.video_size)), # 288 512
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.not_norm_preprocess = T.Compose([
            ToTensorVideo(),
            Resize_Preprocess(tuple(args.video_size))
        ])
        

    def __str__(self):
        return f"{len(self.ann_files)} samples from {self.data_path}"

    def _init_anns(self, data_dir):
        ann_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
        return ann_files

    def _init_sequences(self, ann_files):
        samples = []
        with ThreadPoolExecutor(32) as executor:
            future_to_ann_file = {executor.submit(self._load_and_process_ann_file, ann_file): ann_file for ann_file in ann_files}
            for future in tqdm(as_completed(future_to_ann_file), total=len(ann_files)):
                samples.extend(future.result())
        return samples

    def _load_and_process_ann_file(self, ann_file):

        samples = []
        try:
            with open(ann_file, "r") as f:
                ann = json.load(f)
        except:
            print(f'skip {ann_file}')
            return samples
        n_frames = len(ann['state'])
        for frame_i in range(0, n_frames, self.start_frame_interval):
            sample = dict()
            sample['ann_file'] = ann_file
            sample['frame_ids'] = []
            curr_frame_i = frame_i
            while True:
                if curr_frame_i > (n_frames - 1):
                    break
                sample['frame_ids'].append(curr_frame_i)
                if len(sample['frame_ids']) == self.sequence_length:
                    break
                curr_frame_i += self.sequence_interval
            # make sure there are sequence_length number of frames
            if len(sample['frame_ids']) == self.sequence_length:
                samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_video(self, video_path, frame_ids):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        assert (np.array(frame_ids) < len(vr)).all()
        assert (np.array(frame_ids) >= 0).all()
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        return frame_data
    
    def _load_tokenized_video(self, video_path, frame_ids):
        with open(video_path,'rb') as file:
            video_tensor = torch.load(file)
        # vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        try:
            assert (np.array(frame_ids) < video_tensor.size()[0]).all()
            assert (np.array(frame_ids) >= 0).all()
        except:
            assert False
        frame_data = video_tensor[frame_ids]
        return frame_data
    
    def _get_frames(self, label, frame_ids, cam_id, pre_encode):
        if pre_encode:
            video_path = label['latent_videos'][cam_id]['latent_video_path']
            video_path = os.path.join(self.video_path,video_path)
            frames = self._load_tokenized_video(video_path, frame_ids)
        else:
            video_path = label['videos'][cam_id]['video_path']
            video_path = os.path.join(self.video_path,video_path)
            frames = self._load_video(video_path, frame_ids)
            frames = frames.astype(np.uint8)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2) # (l, c, h, w)

            def printvideo(videos,filename):
                t_videos = rearrange(videos, 'f c h w -> f h w c')
                t_videos = ((t_videos / 2.0 + 0.5).clamp(0, 1) * 255).detach().to(dtype=torch.uint8).cpu().contiguous().numpy()
                print(t_videos.shape)
                writer = imageio.get_writer(filename, fps=4) # fps 是帧率
                for frame in t_videos:
                    writer.append_data(frame) # 1 4 13 23 # fp16 24 76 456 688

            if self.args.normalize:
                frames = self.preprocess(frames)
            else:
                frames = self.not_norm_preprocess(frames)
                frames = torch.clamp(frames*255.0,0,255).to(torch.uint8)
        return frames

    def _get_obs(self, label, frame_ids, cam_id, pre_encode):
        if cam_id is None:
            temp_cam_id = random.choice(self.cam_ids)
        else:
            temp_cam_id = cam_id
        frames = self._get_frames(label, frame_ids, cam_id = temp_cam_id, pre_encode = pre_encode)
        return frames, temp_cam_id

    def _get_robot_states(self, label, frame_ids):
        all_states = np.array(label['state'])
        all_cont_gripper_states = np.array(label['continuous_gripper_state'])
        states = all_states[frame_ids]
        cont_gripper_states = all_cont_gripper_states[frame_ids]
        arm_states = states[:, :6]
        assert arm_states.shape[0] == self.sequence_length
        assert cont_gripper_states.shape[0] == self.sequence_length
        return arm_states, cont_gripper_states

    def _get_all_robot_states(self, label, frame_ids):
        all_states = np.array(label['state'])
        all_cont_gripper_states = np.array(label['continuous_gripper_state'])
        states = all_states[frame_ids]
        cont_gripper_states = all_cont_gripper_states[frame_ids]
        arm_states = states[:, :6]
        return arm_states, cont_gripper_states
    
    def _get_all_actions(self, arm_states, gripper_states, accumulate_action):
        action_num = arm_states.shape[0]-1
        action = np.zeros((action_num, self.action_dim))
        if accumulate_action:
            first_xyz = arm_states[0, 0:3]
            first_rpy = arm_states[0, 3:6]
            first_rotm = euler2rotm(first_rpy)
            for k in range(1, action_num+1):
                curr_xyz = arm_states[k, 0:3]
                curr_rpy = arm_states[k, 3:6]
                curr_gripper = gripper_states[k]
                curr_rotm = euler2rotm(curr_rpy)
                rel_xyz = np.dot(first_rotm.T, curr_xyz - first_xyz)
                rel_rotm = first_rotm.T @ curr_rotm
                rel_rpy = rotm2euler(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = curr_gripper
        else:
            for k in range(1, action_num+1):
                prev_xyz = arm_states[k - 1, 0:3]
                prev_rpy = arm_states[k - 1, 3:6]
                prev_rotm = euler2rotm(prev_rpy)
                curr_xyz = arm_states[k, 0:3]
                curr_rpy = arm_states[k, 3:6]
                curr_gripper = gripper_states[k]
                curr_rotm = euler2rotm(curr_rpy)
                rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
                rel_rotm = prev_rotm.T @ curr_rotm
                rel_rpy = rotm2euler(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = curr_gripper
        return torch.from_numpy(action)  # (l - 1, act_dim)

    def _get_actions(self, arm_states, gripper_states, accumulate_action):
        action = np.zeros((self.sequence_length-1, self.action_dim)) 
        if accumulate_action:
            first_xyz = arm_states[0, 0:3]
            first_rpy = arm_states[0, 3:6]
            first_rotm = euler2rotm(first_rpy)
            for k in range(1, self.sequence_length):
                curr_xyz = arm_states[k, 0:3]
                curr_rpy = arm_states[k, 3:6]
                curr_gripper = gripper_states[k]
                curr_rotm = euler2rotm(curr_rpy)
                rel_xyz = np.dot(first_rotm.T, curr_xyz - first_xyz)
                rel_rotm = first_rotm.T @ curr_rotm
                rel_rpy = rotm2euler(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = curr_gripper
        else:
            for k in range(1, self.sequence_length):
                prev_xyz = arm_states[k - 1, 0:3]
                prev_rpy = arm_states[k - 1, 3:6]
                prev_rotm = euler2rotm(prev_rpy)
                curr_xyz = arm_states[k, 0:3]
                curr_rpy = arm_states[k, 3:6]
                curr_gripper = gripper_states[k]
                curr_rotm = euler2rotm(curr_rpy)
                rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
                rel_rotm = prev_rotm.T @ curr_rotm
                rel_rpy = rotm2euler(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = curr_gripper
        return torch.from_numpy(action)  # (l - 1, act_dim)

    def __getitem__(self, index, cam_id = None, return_video = False):
        if self.mode != 'train':
            np.random.seed(index)
            random.seed(index)

        try:
            sample = self.samples[index]
            ann_file = sample['ann_file']
            frame_ids = sample['frame_ids']
            with open(ann_file, "r") as f:
                label = json.load(f)
            arm_states, gripper_states = self._get_robot_states(label, frame_ids)
            actions = self._get_actions(arm_states, gripper_states, self.accumulate_action)
            actions *= self.c_act_scaler

            data = dict()
            data['action'] = actions.float()

            if self.args.pre_encode:
                latent, cam_id = self._get_obs(label, frame_ids, cam_id, pre_encode=True)
                data['latent'] = latent.float()
                if return_video:
                    video, cam_id = self._get_obs(label, frame_ids, cam_id, pre_encode=False)
                    data['video'] = video.float()
            else:
                video, cam_id = self._get_obs(label, frame_ids, cam_id, pre_encode=False)
                data['video'] = video.float()

            data['video_name'] = {'episode_id': label['episode_id'], 'start_frame_id': str(frame_ids[0]),'cam_id':str(cam_id)}
            return data
        except Exception:
            warnings.warn(f"Invalid data encountered: {self.samples[index]['ann_file']}. Skipped "
                          f"(by randomly sampling another sample in the same dataset).")
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            self.wrong_number += 1
            print(self.wrong_number)
            return self[np.random.randint(len(self.samples))]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/evaluation/bridge/frame_ada.yaml")
    args = parser.parse_args()
    data_config = OmegaConf.load("configs/base/data.yaml")
    diffusion_config = OmegaConf.load("configs/base/diffusion.yaml")
    args = OmegaConf.load(args.config)
    args = OmegaConf.merge(data_config, args)
    args = OmegaConf.merge(diffusion_config, args)
    update_paths(args)
    dataset = Dataset_3D(args,mode='test')

    data_loader = DataLoader(dataset=dataset, 
                                    batch_size=64, 
                                    shuffle=False, 
                                    num_workers=64)
    for data in tqdm(data_loader,total=len(data_loader)):
        pass
    # print(dataset.skip_ann)
        print(data['video'].size())
        print(data['action'].size())

# if __name__ == "__main__":
#     data_dir = "/mnt/bn/robotics-lq2024/gr_data/anns/RT-1/240331/val"
#     sequence_length = 16
#     dataset = Dataset_3D(
#         data_dir,
#         sequence_length,
#         input_size=256,
#         sequence_interval=1,
#         cam_ids=[0],
#         accumulate_action=False,
#         is_training=True,
#     )

#     import matplotlib.pyplot as plt
#     for data_i in range(0, 100000, 10):
#         data = dataset[data_i]
#         rgbs = data['rgbs']
#         actions = data['actions']

#         fig, ax = plt.subplots(sequence_length // 4, 4)
#         for i in range(sequence_length):
#             temp_rgb = rgbs[i].permute(1, 2, 0).numpy()
#             temp_rgb = temp_rgb.astype(np.uint8)
#             ax[i // 4, i % 4].imshow(temp_rgb)
#             ax[i // 4, i % 4].set_axis_off()
#         plt.tight_layout()
#         plt.savefig("debug.png", dpi=300)

#         np.set_printoptions(3, suppress=True)
#         print("----------action----------")
#         print(actions.numpy())

#         arm_states = data['arm_states']
#         gripper_states = data['gripper_states']
#         frame_ids = data['frame_ids']
        
#         import pdb; pdb.set_trace()

    