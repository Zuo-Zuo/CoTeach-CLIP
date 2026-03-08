import os
import shutil
from tqdm import tqdm

# ================= 配置区域 =================
# 请确保这些路径与你刚才运行输出的路径一致
SOURCE_ROOT = './data/rendering'  
SPLIT_FILE = './data/ShapeNet55/ShapeNet-55/train.txt'
TARGET_ROOT = './data/shapenet_moe_ready' 
# ===========================================

TAXONOMY_MAP = {
    '02691156': 'airplane', '02747177': 'trash bin', '02773838': 'bag',
    '02801938': 'basket', '02808440': 'bathtub', '02818832': 'bed',
    '02828884': 'bench', '02843684': 'birdhouse', '02871439': 'bookshelf',
}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_caption(taxonomy_id):
    class_name = TAXONOMY_MAP.get(taxonomy_id, "3d object")
    return f"a 3d rendering of a {class_name}, white background."

def main():
    print(f"=== 开始数据准备 (Fix版) ===")
    print(f"源数据路径: {os.path.abspath(SOURCE_ROOT)}")
    
    if not os.path.exists(SOURCE_ROOT):
        print(f"[错误] 源数据根目录不存在: {SOURCE_ROOT}")
        return

    try:
        with open(SPLIT_FILE, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"[错误] 找不到列表文件: {SPLIT_FILE}")
        return

    # 1. 路径检查逻辑修正
    first_line = lines[0].strip()
    parts = first_line.split('-')
    # 【关键修正】这里增加了 split('.')[0] 去除 .npy 后缀
    test_taxonomy_id = parts[0]
    test_model_id = parts[1].split('.')[0]
    
    test_folder_name = f"{test_taxonomy_id}_{test_model_id}"
    test_src_path = os.path.join(SOURCE_ROOT, test_folder_name)
    
    print(f"--- 路径检查示例 ---")
    print(f"解析后的ID: {test_taxonomy_id} - {test_model_id}")
    print(f"尝试寻找文件夹: {test_src_path}")
    
    if os.path.exists(test_src_path):
        print(f"[成功] 找到了示例文件夹！逻辑已修复。")
    else:
        print(f"[失败] 依然找不到文件夹。请检查上面的解析ID是否正确。")
        # 如果还不行，列出实际目录帮助排查
        try:
            print(f"实际存在的文件夹示例: {os.listdir(SOURCE_ROOT)[:3]}")
        except:
            pass
        return

    # 2. 正式执行
    print(f"正在处理 {len(lines)} 个物体...")
    success_count = 0
    skip_count = 0
    
    for line in tqdm(lines):
        line = line.strip()
        parts = line.split('-')
        if len(parts) < 2: continue
        
        taxonomy_id = parts[0]
        # 【关键修正】去除后缀
        model_id = parts[1].split('.')[0]
        
        src_folder_name = f"{taxonomy_id}_{model_id}"
        src_path = os.path.join(SOURCE_ROOT, src_folder_name)
        dst_folder = os.path.join(TARGET_ROOT, taxonomy_id)
        
        if not os.path.exists(src_path):
            skip_count += 1
            continue

        ensure_dir(dst_folder)
        
        for view_idx in range(10):
            img_filename = f"{view_idx}.png"
            src_img_path = os.path.join(src_path, img_filename)
            
            if not os.path.exists(src_img_path):
                continue
                
            new_filename_base = f"{taxonomy_id}_{model_id}_{view_idx}"
            dst_img_path = os.path.join(dst_folder, new_filename_base + ".png")
            dst_txt_path = os.path.join(dst_folder, new_filename_base + ".txt")
            
            # 创建软链接
            if not os.path.exists(dst_img_path):
                try:
                    os.symlink(os.path.abspath(src_img_path), dst_img_path)
                except OSError:
                    shutil.copy(src_img_path, dst_img_path)
            
            # 生成文本
            if not os.path.exists(dst_txt_path):
                caption = generate_caption(taxonomy_id)
                with open(dst_txt_path, 'w') as f:
                    f.write(caption)
        
        success_count += 1

    print(f"=== 处理完成 ===")
    print(f"成功处理物体数: {success_count}")
    print(f"源文件夹缺失数: {skip_count}")
    print(f"数据已准备在: {TARGET_ROOT}")

if __name__ == "__main__":
    main()