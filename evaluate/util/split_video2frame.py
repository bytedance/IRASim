import os
import imageio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def extract_frames_from_video(video_file, output_folder):
    """
    Extracts all frames from a single MP4 video file and saves them as images in the output folder.

    Args:
        video_file (Path): Path to the MP4 video file.
        output_folder (Path): Path to the folder where the extracted frames will be saved.
    """
    reader = imageio.get_reader(video_file)

    # Create a subfolder for each video file to store its frames
    video_output_folder = output_folder / video_file.stem
    video_output_folder.mkdir(parents=True, exist_ok=True)

    for frame_number, frame in enumerate(reader):
        if frame_number == 0:
            continue
        frame_file = video_output_folder / f"{frame_number:06d}.png"
        imageio.imwrite(frame_file, frame)

    reader.close()

def extract_all_frames_from_videos(input_folder, output_folder):
    """
    Extracts all frames from all MP4 videos in the input folder and saves them as images in the output folder.
    
    Args:
        input_folder (str): Path to the folder containing the MP4 videos.
        output_folder (str): Path to the folder where the extracted frames will be saved.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Get a list of all MP4 files in the input folder
    video_files = list(input_folder.glob('*.mp4'))

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_frames_from_video, video_file, output_folder) for video_file in video_files]
        for future in tqdm(futures, total=len(futures)):
            future.result()

# 