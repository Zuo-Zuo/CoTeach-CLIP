import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import clip
import sys
import os
import numpy as np

# 恢复 Render 模块导入
try:
    from render import Renderer, Selector
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from render.render import Renderer
    from render.selector import Selector

# 导入 CLIP-MoE
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
clip_moe_path = os.path.join(project_root, 'Teacher(MOE)', 'CLIP-MoE-main')
if clip_moe_path not in sys.path:
    sys.path.append(clip_moe_path)

try:
    from clipmoe import model_clipmoe
except ImportError:
    print("Warning: clipmoe not found in path. Trying fallback import...")
    try:
        import model_clipmoe
    except ImportError:
        print("Error: Could not import model_clipmoe.")

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, feat_a, feat_b, temp=None):
        # [MODIFIED] 支持传入动态温度
        t = temp if temp is not None else self.temperature
        logits = torch.matmul(feat_a, feat_b.T) / t
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_a = self.cross_entropy(logits, labels)
        loss_b = self.cross_entropy(logits.T, labels)
        return (loss_a + loss_b) / 2

class CoTeachCLIP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.views = args.views

        self.selector = Selector(args.views, args.dim, args.model)
        self.renderer = Renderer(points_radius=0.02)
        
        # [MODIFIED] 初始化改进方案的参数
        self.log_var_image = nn.Parameter(torch.zeros(1))
        self.log_var_text = nn.Parameter(torch.zeros(1))
        self.log_var_depth = nn.Parameter(torch.zeros(1))
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(0.07)))
        
        self.criterion = ContrastiveLoss()

        print("Initializing CoTeachCLIP Student...")
        standard_clip, _ = clip.load("ViT-B/32", device='cpu')
        self.point_model = deepcopy(standard_clip.visual)
        
        print("Initializing Teacher...")
        moe_ckpt_path = './checkpoints/shapenet-moe-router_weights.pt' 
        if not os.path.exists(moe_ckpt_path):
            moe_ckpt_path = '../CLIP-MoE-main/checkpoints/shapenet-moe-router_weights.pt'
        
        if os.path.exists(moe_ckpt_path):
            try:
                moe_state_dict = torch.load(moe_ckpt_path, map_location='cpu')
                moe_args = [4, 2, 0.0, 12] 
                moe_full_model = model_clipmoe.build_model(moe_state_dict, load_from_clip=False, MoE_args=moe_args)
                moe_full_model.load_state_dict(moe_state_dict, strict=True)
                
                self.image_model = moe_full_model.visual
                self.text_model = moe_full_model 
                
                self.image_model.eval()
                self.text_model.eval()
                for param in self.text_model.parameters():
                    param.requires_grad = False
            except Exception as e:
                print(f"[Error] Load MoE failed: {e}. Using Standard CLIP.")
                self.image_model = deepcopy(standard_clip.visual)
                self.text_model = deepcopy(standard_clip)
        else:
            print("[Warning] Checkpoint not found. Using Standard CLIP.")
            self.image_model = deepcopy(standard_clip.visual)
            self.text_model = deepcopy(standard_clip)

        # 删除了原版的 self.weights 和 self.logit_scale

    def infer(self, points, rot=False):
        azim, elev, dist = self.selector(points)
        imgs = self.renderer(points, azim, elev, dist, self.views, rot=rot)
        b, n, c, h, w = imgs.size()
        imgs = imgs.reshape(b * n, c, h, w)
        imgs = self.point_model(imgs)
        img_feats = imgs / imgs.norm(dim=-1, keepdim=True)
        return img_feats

    def forward(self, points, images, text_tokens, a, e, d):
        # --- Student --- (保持原样)
        depths = self.renderer(points, a, e, d, 1, aug=True, rot=False)
        depth1 = depths[:, 0]
        depth2 = depths[:, 1]
        depth_inputs = torch.cat([depth1, depth2], dim=0)
        depth_feats_all = self.point_model(depth_inputs) 
        depth1_feat, depth2_feat = torch.split(depth_feats_all, points.size(0), dim=0)
        depth_feat = (depth1_feat + depth2_feat) * 0.5
        
        # --- Teacher 1: Image --- (保持原样)
        with torch.no_grad():
            if hasattr(self.image_model, 'conv1'):
                target_dtype = self.image_model.conv1.weight.dtype
            else:
                target_dtype = next(self.image_model.parameters()).dtype
            
            image_input = images.squeeze(1).type(target_dtype)
            self.image_model.eval()
            teacher_output = self.image_model(image_input)
            if isinstance(teacher_output, tuple):
                image_feat = teacher_output[0]
            else:
                image_feat = teacher_output
            image_feat = image_feat.detach().float()

        # --- Teacher 2: Text --- (保持原样)
        with torch.no_grad():
            text_tokens = text_tokens.to(images.device)
            self.text_model.eval()
            text_res = self.text_model.encode_text(text_tokens)
            if isinstance(text_res, tuple):
                text_feat = text_res[0]
            else:
                text_feat = text_res
            text_feat = text_feat.detach().float()

        # --- Norm --- (保持原样)
        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)
        depth_feat = F.normalize(depth_feat, dim=-1)
        depth1_feat = F.normalize(depth1_feat, dim=-1)
        depth2_feat = F.normalize(depth2_feat, dim=-1)

        # === [NEW] 监控相似度分布 ===
        with torch.no_grad():
            # 计算点云与图像的余弦相似度矩阵 [Batch, Batch]
            sim_matrix_img = torch.matmul(depth_feat, image_feat.T)
            # 统计信息（仅取对角线即正样本对，或取整个矩阵）
            # 这里建议监控整个矩阵的分布，反映对比学习的辨别力
            sim_stats = {
                'img_min': sim_matrix_img.min().item(),
                'img_mean': sim_matrix_img.mean().item(),
                'img_max': sim_matrix_img.max().item(),
            }

        # [MODIFIED] 改进损失计算逻辑
        # 1. 约束
        log_var_image = torch.clamp(self.log_var_image, min=-1.0, max=1.0)
        log_var_text = torch.clamp(self.log_var_text, min=-1.0, max=1.0)
        log_var_depth = torch.clamp(self.log_var_depth, min=-1.0, max=1.0)
        temperature = torch.clamp(torch.exp(self.log_temperature), min=0.007, max=0.5)

        # 2. 计算各分支损失
        loss_image = self.criterion(depth_feat, image_feat, temperature)
        loss_text = self.criterion(depth_feat, text_feat, temperature)
        loss_depth = self.criterion(depth1_feat, depth2_feat, temperature)

        # 3. 最终总损失公式
        total_loss = (0.5 / log_var_image.exp() * loss_image + log_var_image +
                      0.5 / log_var_text.exp() * loss_text + log_var_text +
                      0.5 / log_var_depth.exp() * loss_depth + log_var_depth)
        
        return total_loss, loss_image, loss_text, loss_depth, sim_stats