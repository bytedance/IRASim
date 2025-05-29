#!/usr/bin/env python3
import argparse
import subprocess
import os
import tempfile
import datetime

def cat_videos(video_paths, output_dir):
    """
    使用 ffmpeg 将多个视频按顺序拼接成一个视频
    
    Args:
        video_paths: 输入视频路径列表
        output_dir: 输出视频的目录
    """
    # 创建一个临时文件，用于存储视频列表
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        concat_list_path = f.name
        for video_path in video_paths:
            # 使用绝对路径以避免路径问题
            abs_path = os.path.abspath(video_path)
            f.write(f"file '{abs_path}'\n")
    
    # 生成输出文件名（使用时间戳避免重名）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"concatenated_{timestamp}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    
    # 构建 ffmpeg 命令
    cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',  # 允许使用绝对路径
        '-i', concat_list_path,
        '-c', 'copy',  # 直接复制流，不重新编码（更快）
        output_path
    ]
    
    # 执行 ffmpeg 命令
    try:
        subprocess.run(cmd, check=True)
        print(f"视频拼接成功！输出路径: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"拼接视频时出错: {e}")
    finally:
        # 删除临时文件
        os.unlink(concat_list_path)

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='将多个视频按顺序拼接成一个视频')
    parser.add_argument('--video_paths', nargs='+', required=True, 
                        help='输入视频文件的路径列表')
    parser.add_argument('--output_dir', default='.', 
                        help='输出视频的保存目录 (默认: 当前目录)')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 执行视频拼接
    cat_videos(args.video_paths, args.output_dir)

if __name__ == '__main__':
    main()