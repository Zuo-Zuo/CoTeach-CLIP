import os
import h5py
from torch.utils.data import Dataset
import random
import torch
import numpy as np

from datasets.utils import pc_normalize


class ScanObjectNN(Dataset):
    def __init__(self, partition='test', few_num=0, num_points=1024):
        assert partition in ('test', 'training')
        self._load_ScanObjectNN(partition)
        self.num_points = num_points
        self.partition = partition
        self.few_num = few_num
        self._preprocess()        

    def __getitem__(self, item):
        point = self.points[item]
        label = self.labels[item]

        # [修复] 强制转换为 Tensor
        # 因为 pc_normalize 内部使用了 torch.mean，不支持 numpy
        if isinstance(point, np.ndarray):
            point = torch.from_numpy(point)

        # 归一化
        point = pc_normalize(point)
        
        return point, label
    
    """
    def _load_ScanObjectNN(self, partition):
        # -----------------------------------------
        # 第一轮：S-OBJ ONLY (main_split_nobg)
        # -----------------------------------------       
        # 1. 基础目录指向 h5_files
        BASE_DIR = './data/ScanObjectNN/h5_files/'        
        # 2. 指向无背景文件夹
        DATA_DIR = os.path.join(BASE_DIR, 'main_split_nobg')        
        # 3. 文件名 (标准名)
        h5_name = os.path.join(DATA_DIR, f'{partition}_objectdataset.h5')

        if not os.path.exists(h5_name):
            print(f"Error: 找不到 S-OBJ ONLY 数据: {h5_name}")
            raise FileNotFoundError(h5_name)

        print(f"Loading S-OBJ ONLY from: {h5_name}")

        self.points = []
        self.labels = []
        with h5py.File(h5_name, 'r') as f:
            self.points = f['data'][:].astype('float32')
            self.labels = f['label'][:].astype('int64')
        """
    """
    def _load_ScanObjectNN(self, partition):
        # -----------------------------------------
        # 第二轮：S-OBJ BG (main_split / standard file)
        # -----------------------------------------
        
        # 指向 h5_files (或者您存放 main_split 的上级目录)
        BASE_DIR = './data/ScanObjectNN/h5_files/'
        
        # 进入 main_split
        DATA_DIR = os.path.join(BASE_DIR, 'main_split')
        
        # 读取标准文件名 (通常是带背景的)
        h5_name = os.path.join(DATA_DIR, f'{partition}_objectdataset.h5')

        if not os.path.exists(h5_name):
            print(f"Error: 找不到 S-OBJ BG 数据: {h5_name}")
            raise FileNotFoundError(h5_name)
            
        print(f"Loading S-OBJ BG from: {h5_name}")

        self.points = []
        self.labels = []
        with h5py.File(h5_name, 'r') as f:
            self.points = f['data'][:].astype('float32')
            self.labels = f['label'][:].astype('int64')
    """
    #"""
    def _load_ScanObjectNN(self, partition):
        # -----------------------------------------
        # 第三轮：S-PB T50 RS (Augmented Variant)
        # -----------------------------------------
        
        # 指向 h5_files
        BASE_DIR = './data/ScanObjectNN/h5_files/'
        DATA_DIR = os.path.join(BASE_DIR, 'main_split')
        
        # 指定特殊后缀的文件 (PB_T50_RS)
        # 文件名: test_objectdataset_augmentedrot_scale75.h5
        variant_suffix = '_augmentedrot_scale75'
        h5_name = os.path.join(DATA_DIR, f'{partition}_objectdataset{variant_suffix}.h5')

        if not os.path.exists(h5_name):
            print(f"Error: 找不到 S-PB T50 RS 数据: {h5_name}")
            raise FileNotFoundError(h5_name)
            
        print(f"Loading S-PB T50 RS from: {h5_name}")

        self.points = []
        self.labels = []
        with h5py.File(h5_name, 'r') as f:
            self.points = f['data'][:].astype('float32')
            self.labels = f['label'][:].astype('int64')
    #"""
    def _preprocess(self):
        if self.partition == 'training' and self.few_num > 0:
            num_dict = {i: 0 for i in range(15)}
            self.few_points = []
            self.few_labels = []
            random_list = [k for k in range(len(self.labels))]
            random.shuffle(random_list)
            for i in random_list:
                label = self.labels[i].item()
                if num_dict[label] == self.few_num:
                    continue
                self.few_points.append(self.points[i])
                self.few_labels.append(self.labels[i])
                num_dict[label] += 1
        else:
            self.few_points = self.points
            self.few_labels = self.labels

    def __len__(self):
        return len(self.few_labels)
