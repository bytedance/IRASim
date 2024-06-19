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


datasets = ['bridge','rt1','languagetable']
methods = ['frame_ada','video_ada','lvdm']

for dataset in datasets:
    for method in methods:
        commands = []
        gpu_num = 8
        per_gpu_thread = 2
        num_thread = gpu_num * per_gpu_thread
        thread = 0
        for _ in range(per_gpu_thread):
            for rank in range(gpu_num):
                command = f"nohup python3 evaluate/generate_long_video.py --config 'configs/evaluation/{dataset}/{method}.yaml' --rank {rank} --thread {thread} --thread-num {num_thread} > {dataset}_{method}_{thread}.txt 2>&1 &"
                commands.append(command)
                commands.append('sleep 10')
                thread += 1

        commands_joined = "\n".join(commands)

        with open(f'./scripts/generate_long_video_{dataset}_{method}.sh', 'w') as file:
            print(commands_joined,file=file)

