

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
    pred_latent_dir = os.path.join(pred_base_dir,'checkpoints',train_step, f'{args.mode}_sample_latents')
    pred_video_dir = os.path.join(pred_base_dir,'checkpoints',train_step, f'{args.mode}_sample_videos')
    pred_frame_dir = os.path.join(pred_base_dir,'checkpoints',train_step, f'{args.mode}_sample_frames')

    latent_l2 = 0
    psnr = 0
    ssim = 0
    fid = 0
    fvd = 0

    latent_l2 = compute_latent_l2(pred_latent_dir, args.true_sample_latent_videos_dir)
    print(f'pred_latent_dir: {pred_latent_dir}')
    print(f"Model: {model}; Dataset: {args.dataset}; Latent L2: {latent_l2:.4f}")

    psnr, ssim = compute_psnr_ssim(pred_video_dir, args.true_sample_videos_dir)
    print(f'pred_video_dir: {pred_video_dir}')
    print(f"Model: {model}; Dataset: {args.dataset}; PSNR: {psnr:.3f}")
    print(f"Model: {model}; Dataset: {args.dataset}; SSIM {ssim:.3f}")

    # FID command
    fid_project_dir = os.path.join(args.project_dir, "pytorch-fid")
    fid_script_path = "src/pytorch_fid/fid_score.py"
    fid_command = f'cd {fid_project_dir} ; python3 {fid_script_path}  {pred_frame_dir} {args.fid_cache_path}'

    # FVD command
    fvd_project_dir = os.path.join(args.project_dir, "stylegan-v")
    fvd_script_path = "src/scripts/calc_metrics_for_dataset.py"
    fvd_command = f'cd {fvd_project_dir} ; python3 {fvd_script_path} --real_data_path {args.true_sample_frames_dir} --fake_data_path {pred_frame_dir} --mirror 1 --gpus 8 --resolution 256 --metrics fvd_15f --verbose 0 --use_cache 0'
    
    print(f'fid_command: {fid_command}')
    result = subprocess.run(fid_command, shell=True, capture_output=True, text=True)
    # print(result.stdout)
    fid = float(result.stdout.split('FID:')[-1].strip())
    print(f"Model: {model}; Dataset: {args.dataset}; FID: {fid:.2f}")
    
    print(f'fvd_command: {fvd_command}')
    result = subprocess.run(fvd_command, shell=True, capture_output=True, text=True)
    # print(result.stdout)
    pattern = re.compile(r'\{"results":.*?\}')
    match = pattern.search(result.stdout)

    if match:
        json_str = match.group() + '}'
        json_obj = json.loads(json_str)
    else:
        assert False
    fvd = json_obj['results']['fvd_15f']

    print(f"Model: {model}; Dataset: {args.dataset}; FVD: {fvd:.2f}")

    return latent_l2, psnr, ssim, fid, fvd


if __name__ == "__main__":
    models = ['vdm', 'lvdm', 'video_ada', 'frame_ada']
    datasets = ['rt1', 'bridge', 'languagetable']
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
            latent_l2, psnr, ssim, fid, fvd = auto_evaluate(args,model)
            results.append([dataset, model, latent_l2, psnr, ssim, fid, fvd])

    df = pd.DataFrame(results, columns=['Dataset', 'Method', 'Latent L2', 'PSNR', 'SSIM', 'FID', 'FVD'])
    df['Latent L2'] = df['Latent L2'].round(4)
    df['PSNR'] = df['PSNR'].round(3)
    df['SSIM'] = df['SSIM'].round(3)
    df['FID'] = df['FID'].round(2)
    df['FVD'] = df['FVD'].round(2)

    print(df.to_string(index=False))

   


