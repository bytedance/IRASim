

import os
import re
import json
import subprocess
import pandas as pd
from omegaconf import OmegaConf

from util import update_paths
from evaluate.compute_latent_l2 import compute_latent_l2
from evaluate.compute_psnr_ssim import compute_psnr_ssim, compute_psnr

def auto_evaluate(args,model):
    update_paths(args)

    pred_date = '06/14'
    pred_base_dir = os.path.join(args.results_dir, pred_date, f'{args.mode}_{args.dataset}_{model}-debug')
    train_step = args.evaluate_checkpoint.split('/')[-1].split('.')[0]
    pred_latent_dir = os.path.join(pred_base_dir,'checkpoints',train_step, f'{args.mode}_episode_latent_videos')
    pred_video_dir = os.path.join(pred_base_dir,'checkpoints',train_step, f'{args.mode}_episode_videos')

    latent_l2 = compute_latent_l2(pred_latent_dir, args.true_episode_latent_videos_dir)
    print(f'pred_latent_dir: {pred_latent_dir}')
    print(f"Model: {model}; Dataset: {args.dataset}; Latent L2: {latent_l2:.4f}")

    psnr = compute_psnr(pred_video_dir, args.true_episode_videos_dir)
    print(f'pred_video_dir: {pred_video_dir}')
    print(f"Model: {model}; Dataset: {args.dataset}; PSNR: {psnr:.3f}")
    return latent_l2,psnr


if __name__ == "__main__":
    models = ['lvdm', 'video_ada' , 'frame_ada']
    datasets = ['rt1','bridge','languagetable']
    models = ['frame_ada']
    datasets = ['bridge']
    results = []
    for dataset in datasets:
        for model in models:
            config_path = f"configs/evaluation/{dataset}/{model}.yaml"
            data_config = OmegaConf.load("configs/base/data.yaml")
            diffusion_config = OmegaConf.load("configs/base/diffusion.yaml")
            config = OmegaConf.load(config_path)
            config = OmegaConf.merge(data_config, config)
            config = OmegaConf.merge(diffusion_config, config)
            args = config
            latent_l2, psnr = auto_evaluate(args,model)
            results.append([dataset, model, latent_l2, psnr])

    df = pd.DataFrame(results, columns=['Dataset', 'Method', 'Latent L2', 'PSNR'])
    df['Latent L2'] = df['Latent L2'].round(4)
    df['PSNR'] = df['PSNR'].round(3)

    print(df.to_string(index=False))

            

