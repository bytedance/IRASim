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

from dataset.dataset_2D import Dataset_2D
from dataset.dataset_3D import Dataset_3D

def get_dataset(args):
    if args.dataset == 'languagetable':
        if args.do_evaluate:
            return None, Dataset_2D(args, mode=args.mode)
        elif args.debug:
            return Dataset_2D(args,mode='val'), Dataset_2D(args, mode='val')
        else:
            return Dataset_2D(args,mode='train'), Dataset_2D(args,mode='val')
    elif args.dataset == 'rt1' or  args.dataset == 'droid' or  args.dataset == 'bridge':
        if args.do_evaluate:
            return None, Dataset_3D(args,mode=args.mode)
        elif args.debug:
            return Dataset_3D(args,mode='val'), Dataset_3D(args,mode='val')
        else:
            return Dataset_3D(args,mode='train'), Dataset_3D(args,mode='val')
    else:
        raise NotImplementedError(args.dataset)