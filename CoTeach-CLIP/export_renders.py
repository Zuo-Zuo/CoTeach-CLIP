import os
import torch
from datasets.shapenet import ShapeNetRender
from torchvision.utils import save_image
from tqdm import tqdm

def export_shapenet_for_mcl(root_dir, output_dir):
    # 初始化数据集，split='train'
    # 注意：我们需要所有视角，不仅仅是随机视角。
    # 这里我们简单实例化数据集，然后手动遍历
    dataset = ShapeNetRender(root_dir, split='train')
    
    print(f"Total objects: {len(dataset.shape_names)}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ShapeNetRender 的文件结构本身就是 synset_id/name/render_xx.png
    # 我们不需要重新渲染，只需要按照 CLIP-MoE 能够读取的 ImageFolder 格式整理链接或复制
    # 为了节省空间，建议使用软链接 (Symlink)
    
    for idx, name in enumerate(tqdm(dataset.shape_names)):
        synset_id = dataset.synset_ids[idx]
        # 原始路径假设
        # dataset.render_path = root_dir/ShapeNetRender
        src_folder = os.path.join(dataset.render_path, synset_id, name)
        
        # 目标文件夹
        dst_folder = os.path.join(output_dir, synset_id)
        os.makedirs(dst_folder, exist_ok=True)
        
        # 遍历 10 个固定视角 (00.png 到 09.png)
        # dataset.views_azim 定义了这些视角
        for i in range(10): # 假设有10个视角
            filename = f"render_{i:02d}.png" # 或者是 render00.png，需根据实际文件检查
            # 或者是根据 dataset.views_azim 这里的逻辑，查看 shapenet.py 的 __getitem__
            # 原代码: image_path = os.path.join(self.render_path, shape_id, shape_name, 'render_%02d.png' % rand_idx)
            
            src_path = os.path.join(src_folder, f"render_{i:02d}.png")
            dst_path = os.path.join(dst_folder, f"{name}_v{i:02d}.png")
            
            if os.path.exists(src_path):
                if not os.path.exists(dst_path):
                    os.symlink(os.path.abspath(src_path), dst_path)

if __name__ == "__main__":
    # 修改这里的路径
    SHAPENET_ROOT = "./data/ShapeNet" # 指向你的原始 ShapeNet 路径
    OUTPUT_ROOT = "./ShapeNet_MCL_Export"
    export_shapenet_for_mcl(SHAPENET_ROOT, OUTPUT_ROOT)