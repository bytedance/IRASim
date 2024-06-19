# sudo apt-get install ffmpeg libsm6 libxext6  -y
# https://pytorch.org/get-started/locally/ 
# install torch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install timm diffusers[torch]==0.24.0 einops transformers scikit-image decord pandas imageio-ffmpeg omegaconf huggingface_hub nvitop deepspeed matplotlib opencv-python wandb rotary_embedding_torch einops_exts tensorflow tensorflow_datasets 
# Optionally
# pip3 install flash-attn --no-build-isolation