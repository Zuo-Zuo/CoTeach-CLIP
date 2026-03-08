import os
import torch
import numpy as np
import torch.utils.data as data
import h5py
from typing import Tuple
import collections
from pytorch3d.io import load_obj
import random
from torchvision.transforms import Normalize, ToTensor
from PIL import Image
import sys

from render.render import Renderer

# [Import] 尝试导入 CLIP-MoE 的 tokenize
try:
    from clipmoe.clipmoe import tokenize
except ImportError:
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    clip_moe_path = os.path.join(project_root, 'CLIP-MoE-main')
    if os.path.exists(clip_moe_path):
        sys.path.append(clip_moe_path)
    try:
        from clipmoe.clipmoe import tokenize
    except ImportError:
        print("Warning: Could not import clipmoe.tokenize. Text encoding will fail.")
        def tokenize(text, truncate=True):
             return torch.zeros(1, 77, dtype=torch.long)

# 映射表
TAXONOMY_MAP = {
    '02691156': 'airplane', '02747177': 'trash bin', '02773838': 'bag',
    '02801938': 'basket', '02808440': 'bathtub', '02818832': 'bed',
    '02828884': 'bench', '02843684': 'birdhouse', '02871439': 'bookshelf',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02942699': 'camera', '02946921': 'can',
    '02954340': 'cap', '02958343': 'car', '02992529': 'cellphone',
    '03001627': 'chair', '03046257': 'clock', '03085013': 'keyboard',
    '03207941': 'dishwasher', '03211117': 'display', '03261776': 'earphone',
    '03325088': 'faucet', '03337140': 'file cabinet', '03467517': 'guitar',
    '03513137': 'helmet', '03593526': 'jar', '03624134': 'knife',
    '03636649': 'lamp', '03642806': 'laptop', '03691459': 'loudspeaker',
    '03710193': 'mailbox', '03759954': 'microphone', '03761084': 'microwave',
    '03790512': 'motorcycle', '03797390': 'mug', '03928116': 'piano',
    '03938244': 'pillow', '03948459': 'pistol', '03991062': 'pot',
    '04004475': 'printer', '04074963': 'remote control', '04090263': 'rifle',
    '04099429': 'rocket', '04225987': 'skateboard', '04256520': 'sofa',
    '04330267': 'stove', '04379243': 'table', '04401088': 'telephone',
    '04460130': 'tower', '04468005': 'train', '04530566': 'vessel',
    '04554684': 'washer'
}

cat_labels = {'02691156': 0, '02747177': 1, '02773838': 2, '02801938': 3, '02808440': 4, '02818832': 5, '02828884': 6, '02843684': 7, '02871439': 8, '02876657': 9, '02880940': 10, '02924116': 11, '02933112': 12, '02942699': 13, '02946921': 14, '02954340': 15, '02958343': 16, 
'02992529': 17, '03001627': 18, '03046257': 19, '03085013': 20, '03207941': 21, '03211117': 22, '03261776': 23, '03325088': 24, '03337140': 25, '03467517': 26, '03513137': 27, '03593526': 28, '03624134': 29, '03636649': 30, '03642806': 31, '03691459': 32, '03710193': 33, 
'03759954': 34, '03761084': 35, '03790512': 36, '03797390': 37, '03928116': 38, '03938244': 39, '03948459': 40, '03991062': 41, '04004475': 42, '04074963': 43, '04090263': 44, '04099429': 45, '04225987': 46, '04256520': 47, '04330267': 48, '04379243': 49, '04401088': 50, 
'04460130': 51, '04468005': 52, '04530566': 53, '04554684': 54}


class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)
        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]

# ================= 数据增强函数 (Strong Augmentation) =================
def translate_point_cloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def random_scale_point_cloud(pointcloud, scale_low=0.8, scale_high=1.25):
    scale = np.random.uniform(scale_low, scale_high)
    scaled_pointcloud = np.multiply(pointcloud, scale).astype('float32')
    return scaled_pointcloud

def jitter_point_cloud(pointcloud, sigma=0.01, clip=0.05):
    # 这是模拟真实雷达/深度相机噪声最关键的一步
    N, C = pointcloud.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_pointcloud = pointcloud + jittered_data
    return jittered_pointcloud.astype('float32')

def rotate_perturbation_point_cloud(pointcloud, angle_sigma=0.06, angle_clip=0.18):
    # 微小旋转扰动，增加对视角误差的鲁棒性
    rotated_data = np.zeros(pointcloud.shape, dtype=np.float32)
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
                   [0,np.cos(angles[0]),-np.sin(angles[0])],
                   [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                   [0,1,0],
                   [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                   [np.sin(angles[2]),np.cos(angles[2]),0],
                   [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))
    shape_pc = pointcloud.shape
    rotated_data = np.dot(pointcloud.reshape((-1, 3)), R)
    return rotated_data
# =====================================================================

def torch_center_and_normalize(points,p="inf"):
    N = points.shape[0]
    center = points.mean(0)
    if p != "fro" and p!= "no":
        scale = torch.max(torch.norm(points - center, p=float(p),dim=1))
    elif p=="fro" :
        scale = torch.norm(points - center, p=p )
    elif p=="no":
        scale = 1.0
    points = points - center.expand(N, 3)
    points = points * (1.0 / float(scale))
    return points


class ShapeNet(data.Dataset):
    def __init__(self, partition='train', whole=False, num_points=1024):
        assert partition in ['train', 'test']
        self.data_root = './data/ShapeNet55/ShapeNet-55'
        self.pc_path = './data/ShapeNet55/shapenet_pc'
        self.subset = partition
        self.npoints = 8192
        
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')
        
        self.sample_points_num = num_points
        self.whole = whole

        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            lines = test_lines + lines
        self.file_list = []
        check_list = ['03001627-udf068a6b', '03001627-u6028f63e', '03001627-uca24feec', '04379243-', '02747177-', '03001627-u481ebf18', '03001627-u45c7b89f', '03001627-ub5d972a1', '03001627-u1e22cc04', '03001627-ue639c33f']
        
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]

            if taxonomy_id + '-' + model_id not in check_list:
                self.file_list.append({
                    'taxonomy_id': taxonomy_id,
                    'model_id': model_id,
                    'file_path': line
                })

        self.permutation = np.arange(self.npoints)

    def _load_mesh(self, model_path) -> Tuple:
        verts, faces, aux = load_obj(model_path, create_texture_atlas=True, texture_wrap='clamp')
        textures = aux.texture_atlas    
        return verts, faces.verts_idx, textures

    def pc_norm(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc        

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        points = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        points = self.pc_norm(points)
        points = torch.from_numpy(points).float()
        verts, faces, textures = self._load_mesh(os.path.join('/data/ShapeNetCore.v2', sample['taxonomy_id'], sample['model_id'], 'models', 'model_normalized.obj'))
        verts = torch_center_and_normalize(verts.to(torch.float), '2.0')
        mesh = dict()
        mesh["verts"] = verts
        mesh["faces"] = faces
        mesh["textures"] = textures
        label = sample['taxonomy_id'] + '_' + sample['model_id']
        return points, mesh, label

    def __len__(self):
        return len(self.file_list)


class ShapeNetRender(ShapeNet):
    def __init__(self, partition='train', whole=False, num_points=1024):
        super().__init__(partition, whole, num_points)
        self.partition = partition
        self.views_dist = torch.ones((10), dtype=torch.float, requires_grad=False)
        self.views_elev = torch.asarray((0, 90, 180, 270, 225, 225, 315, 315, 0, 0), dtype=torch.float, requires_grad=False)
        self.views_azim = torch.asarray((0, 0, 0, 0, -45, 45, -45, 45, -90, 90), dtype=torch.float, requires_grad=False)
        self.render = Renderer()
        self.totensor = ToTensor()
        self.norm = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        points = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        
        # 基础采样
        points = self.random_sample(points, self.sample_points_num)
        points = self.pc_norm(points)

        # [Strong Augmentation] 仅在训练时应用
        if self.partition == 'train':
            # 1. 缩放 (模拟物体大小变化)
            points = random_scale_point_cloud(points)
            # 2. 偏移 (模拟对齐误差)
            points = translate_point_cloud(points)
            # 3. 抖动 (模拟雷达/扫描噪声 - 最重要!)
            points = jitter_point_cloud(points, sigma=0.01, clip=0.05)
            # 4. 微小旋转 (模拟视角偏差)
            points = rotate_perturbation_point_cloud(points)

        points = torch.from_numpy(points).float()

        if self.partition == 'test':
            return points, cat_labels[sample['taxonomy_id']]
        
        name = sample['taxonomy_id'] + '_' + sample['model_id']
        rand_idx = random.randint(0, 9)
        
        # Image Teacher: 输入是干净的预渲染图像
        image = Image.open('./data/rendering/%s/%d.png' % (name, rand_idx))
        image = self.norm(self.totensor(image))

        # Text Teacher: 文本描述
        taxonomy_id = sample['taxonomy_id']
        class_name = TAXONOMY_MAP.get(taxonomy_id, "object")
        caption = f"a 3d rendering of a {class_name}"
        text_token = tokenize(caption, truncate=True)[0]

        return image, points, text_token, self.views_azim[rand_idx], self.views_elev[rand_idx], self.views_dist[rand_idx]

def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        return torch.stack(batch, 0)
    elif elem_type.__module__ == 'pytorch3d.structures.meshes':
        return batch
    elif isinstance(elem, dict):
        return batch
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, (int)):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
        return elem_type(*(collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError(
                'each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]