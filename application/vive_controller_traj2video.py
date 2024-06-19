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

from einops import rearrange, repeat
import imageio
import os
import h5py
import argparse
import json
import pickle
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
from einops import rearrange
from dataset.dataset_util import euler2rotm, quat2rotm, rotm2euler
from models import get_models
from diffusion import create_diffusion,create_mask_diffusion
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL,AutoencoderKLTemporalDecoder
from diffusers.optimization import get_scheduler
from dataset import get_dataset
from util import (clip_grad_norm_, create_logger, update_ema, 
                   requires_grad, cleanup, setup_distributed,
                   get_experiment_dir, text_preprocessing)
from sample.pipeline_trajectory2videogen import Trajectory2VideoGenPipeline

from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler, 
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler, 
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)
import numpy as np
import torch


def generate_single_video(args, batch, device, vae, mask_frame_num, model):
    with torch.no_grad():
        x = batch['start_image'].to(device).to(torch.float32)
        actions = batch['action'].to(device).to(torch.float32)
        f = actions.size()[0]+1

        # x = x.unsqueeze(0)
        # x = vae.encode(x).latent_dist.sample().mul_(vae.config.scaling_factor) 

        mask_x = x.unsqueeze(0).unsqueeze(0)
        actions = actions.unsqueeze(0)

        #################
        # denoise x
        #################

        if args.sample_method == 'PNDM':
        if args.sample_method == 'PNDM':
            scheduler = PNDMScheduler.from_pretrained(args.scheduler_path, 
                                                beta_start=args.beta_start, 
                                                beta_end=args.beta_end, 
                                                beta_schedule=args.beta_schedule,
                                                variance_type=args.variance_type)
        elif args.sample_method == 'DDPM':
            scheduler = PNDMScheduler.from_pretrained(args.scheduler_path,
                                                beta_start=args.beta_start, 
                                                beta_end=args.beta_end, 
                                                beta_schedule=args.beta_schedule,
                                                variance_type=args.variance_type)
        else:
            assert False

        videogen_pipeline = Trajectory2VideoGenPipeline(vae=vae, 
                        #  text_encoder=text_encoder, 
                        #  tokenizer=tokenizer, 
                            scheduler=scheduler, 
                            transformer=model) # .to(device)

        # print('Generation total {} videos'.format(1))
        videos, latents = videogen_pipeline(
                                actions, 
                                mask_x = mask_x,
                                video_length=args.num_frames, 
                                height=args.video_size[0], 
                                width=args.video_size[1], 
                                num_inference_steps=args.infer_num_sampling_steps,
                                guidance_scale=args.guidance_scale, # dumpy
                                device = device,
                                return_dict = False
                                )
        # videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255).detach().to(dtype=torch.uint8).cpu().contiguous()
        return videos, latents

def h5_to_json(h5_filename):
    """
    将HDF5文件转换为JSON文件。
    
    :param h5_filename: 输入的HDF5文件名。
    :param json_filename: 输出的JSON文件名。
    """
    with h5py.File(h5_filename, 'r') as h5file:
        result = {}  # 用于存储转换后的数据
        
        def recursive_convert(group, result):
            """
            递归转换HDF5文件内容到字典。
            
            :param group: 当前处理的HDF5组或根文件。
            :param result: 存储转换结果的字典。
            """
            for key, item in group.items():
                if isinstance(item, h5py.Dataset):  # 如果是数据集，直接读取数据
                    result[key] = item[()].tolist()  # 将numpy数组转换为列表
                elif isinstance(item, h5py.Group):  # 如果是组，递归处理
                    result[key] = {}
                    recursive_convert(item, result[key])
                    
        recursive_convert(h5file, result)
        return result

def process_frames_flexibly(frames, vae, batch_size=40):
    with torch.no_grad():
        n_frames = len(frames)
        # 计算需要多少个完整批次，以及最后一个批次需要包含的帧数
        n_full_batches = n_frames // batch_size
        remainder = n_frames % batch_size
        
        # 初始化列表来存储处理后的批次
        processed_batches = []
        
        # 处理完整的批次
        for i in range(n_full_batches):
            batch = frames[i*batch_size : (i+1)*batch_size]
            scaled = vae.decode(batch / vae.config.scaling_factor).sample
            processed_batches.append(scaled)
        
        # 如果有剩余的帧，处理这些帧
        if remainder > 0:
            last_batch = frames[-remainder:] # 获取最后 remainder 帧

            scaled = vae.decode(last_batch / vae.config.scaling_factor).sample

            processed_batches.append(scaled)
        
        # 合并所有处理后的批次
        processed_frames = torch.cat(processed_batches, dim=0)
        return processed_frames

def main(args,rank):
    vive_data = h5_to_json('data/test/data.hdf5')

    # btn = vive_data['btn']['btn_0']
    # vive_actions = vive_data['vive']['vive_0']
    # for i in range(len(vive_actions)):
    #     x,y,z,w = vive_actions[i][3:]
    #     r = R.from_quat([x,y,z,w])
    #     r = r.as_euler('xyz', degrees=False)
    #     vive_actions[i][3:] = r.tolist()
        # x,y,z = 

    # # Debug: find vive axis
    # from dataset.utils import quat2rotm, rotm2euler
    # vive_states = np.array(vive_data['vive']['vive_0'])
    # vive_states = vive_states[:100]
    # n_frames = vive_states.shape[0]
    # print(f"n_frames: {n_frames}")
    # actions = np.zeros((n_frames, 7))
    # first_xyz = vive_states[0, :3]
    # first_rpy = vive_states[0, 3:]
    # first_rotm = quat2rotm(first_rpy)
    # for i in range(0, n_frames):
    #     curr_xyz  = vive_states[i, :3]
    #     curr_quat = vive_states[i, 3:]
    #     curr_rotm = quat2rotm(curr_quat)
    #     delta_xyz = first_rotm.T @ (curr_xyz - first_xyz)
    #     delta_rpy = rotm2euler(first_rotm.T @ curr_rotm)
    #     actions[i, 0:3] = delta_xyz
    #     actions[i, 3:6] = delta_rpy
    #     # action[i, -1] = gripper # TODO: add gripper signal
    # print(actions[:, :3])
    # import pdb; pdb.set_trace()

    # offset rotation
    offset_rotm = np.array([
        [-1,  0, 0],
        [ 0, -1, 0],
        [ 0,  0, 1]
    ])

    vive_states = np.array(vive_data['vive']['vive_0'])
    btn = np.array(vive_data['btn']['btn_0'])
    for i in range(vive_states.shape[0]):
        if np.all(vive_states[i] == 0.0):
            break
    vive_states = vive_states[:i]
    btn = btn[:i]
    n_frames = vive_states.shape[0]
    print(f"n_frames: {n_frames}")
    interval = 5
    actions = np.zeros((n_frames // interval + 1, 7))
    for i in range(0, n_frames - interval, interval):
        curr_xyz = vive_states[i, :3]
        curr_quat = vive_states[i, 3:]
        curr_rotm = quat2rotm(curr_quat)
        next_xyz = vive_states[i + interval, :3]
        next_quat = vive_states[i + interval, 3:]
        next_rotm = quat2rotm(next_quat)
        next_gripper = btn[i+interval][0]

        curr_rotm = curr_rotm @ offset_rotm
        next_rotm = next_rotm @ offset_rotm
        delta_rotm = curr_rotm.T @ next_rotm
        delta_rpy = rotm2euler(delta_rotm)
        delta_xyz = curr_rotm.T @ (next_xyz - curr_xyz)
        actions[i // interval, 0:3] = delta_xyz
        actions[i // interval, 3:6] = delta_rpy
        actions[i // interval, -1] = next_gripper

    gripper = actions[:,-1].tolist()
    gripper = smooth_ones(gripper)
    actions [:,-1] = np.array(gripper)

    np.set_printoptions(4, suppress=True)
    print(actions)
    # import pdb; pdb.set_trace()


    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    if os.path.exists('/mnt/bn/'):
        nas_dir = '/mnt/bn//'
    elif os.path.exists('/mnt/bn/'):
        nas_dir = '/mnt/bn//'
    else:
        nas_dir = '/mnt/bn//'
    args.nas_dir = nas_dir
    args.train_annotation_path = os.path.join(nas_dir+'/',args.train_annotation_path)
    args.val_annotation_path = os.path.join(nas_dir+'/',args.val_annotation_path)
    args.test_annotation_path = os.path.join(nas_dir+'/',args.test_annotation_path)
    args.pretrained_model_path = os.path.join(nas_dir+'/',args.pretrained_model_path)
    args.vae_model_path = os.path.join(nas_dir+'/',args.vae_model_path)
    args.results_dir = os.path.join(nas_dir+'/',args.results_dir)
    args.val_fvd_cache = os.path.join(nas_dir+'/',args.val_fvd_cache)
    args.val_fid_cache = os.path.join(nas_dir+'/',args.val_fid_cache)
    args.test_fvd_cache = os.path.join(nas_dir+'/',args.test_fvd_cache)
    args.test_fid_cache = os.path.join(nas_dir+'/',args.test_fid_cache)
    if args.mode == 'test':
        fid_cache = args.test_fid_cache
        fvd_cache = args.test_fvd_cache
    elif args.mode == 'val':
        fid_cache = args.val_fid_cache
        fvd_cache = args.val_fvd_cache
    if args.resume_from_checkpoint:
        args.resume_from_checkpoint = os.path.join(nas_dir+'/',args.resume_from_checkpoint)
    if args.pretrained:
        args.pretrained = os.path.join(nas_dir+'/',args.pretrained)

    device = torch.device("cuda", 0)
    # device = torch.device("cpu")

    # Create model:
    args.latent_size = [t // 8 for t in args.video_size]
    model = get_models(args)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_mask_diffusion(timestep_respacing="")
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    if args.use_temporal_decoder:
        vae = AutoencoderKLTemporalDecoder.from_pretrained(args.vae_model_path, subfolder="t2v_required_models/vae_temporal_decoder").to(device)
    else:
        vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device)

    auto_video_path = os.path.join('figures/rt1_env',f'auto_video_{args.anno}')
    os.makedirs(auto_video_path,exist_ok=True)
    print(auto_video_path)

    start_frame = 4
    train_dataset,val_dataset = get_dataset(args)
    #val_loader = DataLoader(dataset=val_dataset, batch_size=args.local_val_batch_size, shuffle=False, num_workers=args.num_workers)

    from tqdm import tqdm
    for sample_idx, ann_file in tqdm(enumerate(val_dataset.ann_files),total=len(val_dataset.ann_files)):
        with open(ann_file, "rb") as f:
            ann = json.load(f)
        # if sample_idx == 170:
        #     break
        # if ann['texts'][0] == 'close fridge':
        #     break
        if ann['episode_id'] == '11341': # rt1 4 7# bridge 91 15 # 171
            break
    with open(ann_file, "r") as f:
        ann = json.load(f)
    video_path = ann['videos'][0]['video_path']
    video_reader = imageio.get_reader(video_path)
    video_tensor = []
    for frame in video_reader:
        frame_tensor = torch.tensor(frame)
        video_tensor.append(frame_tensor)
    video_reader.close()
    video_tensor = torch.stack(video_tensor)

    ori_video = video_tensor[start_frame:]

    imageio.imwrite('figures/rt1_env/first_frame.png', ori_video[0])  # 保存为PNG格式


    # # use pretrained model?
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            print('Using ema ckpt!')
            checkpoint = checkpoint["ema"]

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict:
                pretrained_dict[k] = v
            else:
                print('Ignoring: {}'.format(k))
        print('Successfully Load {}% original pretrained model weights '.format(len(pretrained_dict) / len(checkpoint.items()) * 100))
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('Successfully load model at {}!'.format(args.pretrained)) 
    model.to(device)
    model.eval()
    # with open('batch.pickle', 'rb') as f:
    #     batch = pickle.load(f)
    mask_frame_num = args.mask_frame_num


    def printvideo(videos,filename):
        t_videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255).detach().to(dtype=torch.uint8).cpu().contiguous()
        t_videos = rearrange(t_videos, 'f c h w -> f h w c')
        t_videos = t_videos.numpy()
        # filename = f"{videos_dir}/train_steps_{train_steps}.mp4"
        # print(.format(1))
        print('Generate',t_videos.shape, 'shape video')
        writer = imageio.get_writer(filename, fps=4) 
        for frame in t_videos:
            writer.append_data(frame) # 1 4 13 23
        print(filename)

    ori_video = ((ori_video/255.0)-0.5)*2



    tokenize_video_path = ann['tokenized_videos'][0]['tokenized_video_path']
    with open(tokenize_video_path, 'rb') as file:
        ori_video_tokenized = torch.load(file)[start_frame:]
    total_frame = ori_video_tokenized.size()[0]
    # frame_ids = list(range(total_frame))
    # arm_states, gripper_states = val_dataset._get_robot_states(ann, frame_ids)
    # action = val_dataset._get_actions(arm_states, gripper_states, args.accumulate_action)
    actions *= val_dataset.c_act_scaler
    action = torch.from_numpy(actions)
    # action = sample['action']
    

    current_frame = 0
    start_image = ori_video_tokenized[current_frame]
    seg_video_list = []
    seg_idx = 0
    total_frame = action.shape[0]
    while current_frame+args.num_frames-1 < total_frame:
        seg_action = action[current_frame:current_frame+args.num_frames-1]
        batch = {'start_image':start_image,'action':seg_action}
        seg_video, seg_latents = generate_single_video(args, batch,device,vae,mask_frame_num,model)
        seg_video = seg_video.squeeze()
        seg_latents = seg_latents.squeeze()
        start_image = seg_latents[-1].clone()
        current_frame += args.num_frames-1
        seg_video_list.append(seg_video)
        printvideo(seg_video,f'{rank}-{seg_idx}.mp4')
        seg_idx += 1
    
    
    seg_action = action[current_frame:]
    true_action = seg_action.size()[0]
    if true_action != 0:
        false_action = args.num_frames-true_action-1
        seg_false_action = repeat(seg_action[0], 'd -> f d', f= false_action) 
        
        com_action = torch.cat([seg_action,seg_false_action],dim=0)
        batch = {'start_image':start_image,'action':com_action}
        seg_video, seg_latents = generate_single_video(args, batch,device,vae,mask_frame_num,model)
        seg_video = seg_video.squeeze()
        seg_video_list.append(seg_video[0:true_action+1])

    com_video_list = [seg_video_list[0]] + [video[1:] for video in seg_video_list[1:]]
    com_video = torch.cat(com_video_list,dim=0).cpu()
    

    # ori_video = ori_video.to(device)
    # ori_video = process_frames_flexibly(ori_video, vae, batch_size=20)
    # ori_video = ori_video.cpu()

    ori_video = rearrange(ori_video,'f h w c -> f c h w')
    # com_video = torch.cat([ori_video,com_video],dim=2)
    
    printvideo(com_video,os.path.join(auto_video_path, f'{ann["episode_id"]}.mp4'))

    # print(os.path.join(auto_video_path, f'{sample_idx}.mp4'))

def smooth_ones(arr):
    n = len(arr)
    result = [0.0] * n  # 初始化结果数组
    i = 0
    
    while i < n:
        if arr[i] == 1:
            # 找到连续1的开始和结束索引
            start = i
            while i < n and arr[i] == 1:
                i += 1
            end = i
            
            length = end - start
            # 处理长度小于10的情况
            if length < 10:
                mid_point = length // 2
                # 对于小于10个连续1的数组，平滑处理要考虑重叠
                for j in range(length):
                    # 前半段平滑，使用min确保平滑值不超过1.0
                    front_scale = (j + 1) * 0.2 if j < 5 else 1.0
                    # 后半段平滑
                    back_scale = ((length - j) * 0.2) if j >= length - 5 else 1.0
                    result[start + j] = min(front_scale, back_scale)
            else:
                # 前5个平滑值
                for j in range(5):
                    result[start + j] = (j + 1) * 0.2
                # 中间的保持为1.0
                for j in range(5, length - 5):
                    result[start + j] = 1.0
                # 后5个平滑值
                for j in range(length - 5, length):
                    result[start + j] = 1.0 - (j - (length - 5)) * 0.2
        else:
            i += 1
    
    return result


# 示例

def action2traj(actions):
    actions = np.array(actions)
    all_rel_xyz = actions[:, :3]
    all_rel_rpy = actions[:, 3:6]
    print(actions)

    all_xyz = []
    all_xyz.append(np.array([0.0, 0.0, 0.0]))

    n_frames = actions.shape[0]
    curr_xyz = np.array([0.0, 0.0, 0.0])
    curr_rotm = np.array([
        [           0.0, 1.0,            0.0],
        [np.sqrt(2) / 2, 0.0,  np.sqrt(2) / 2],
        [np.sqrt(2) / 2, 0.0, -np.sqrt(2) / 2],
    ])
    for i in range(0, n_frames):
        rel_xyz = all_rel_xyz[i]
        rel_rpy = all_rel_rpy[i]
        rel_rotm = euler2rotm(rel_rpy)
        curr_xyz = curr_rotm @ rel_xyz + curr_xyz
        curr_rotm = curr_rotm @ rel_rotm
        all_xyz.append(curr_xyz)
    return all_xyz

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/exp/rt1_eva/frame_ada.yaml")
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    rank = args.rank
    main(OmegaConf.load(args.config),rank)



    