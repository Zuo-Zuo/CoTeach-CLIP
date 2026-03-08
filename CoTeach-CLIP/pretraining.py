import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip
import torch_optimizer as optim
from torch.utils.tensorboard import SummaryWriter

from models import CoTeachCLIP
from datasets import ModelNet40Align, ShapeNetRender
from utils import IOStream

# 预加载 CLIP
clip_model, _ = clip.load("ViT-B/32", device='cpu')

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

def _init_(path, args):
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path + '/' + args.exp_name):
        os.makedirs(path + '/' + args.exp_name)

def train(args, io):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # --- Validation Zero-shot Prompts ---
    test_prompts = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower pot', 'glass box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night stand', 'person', 'piano', 'plant', 'radio', 'range hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv stand', 'vase', 'wardrobe', 'xbox']
    val_prompts = ['airplane', 'ashcan', 'bag', 'basket', 'bathtub', 'bed', 'bench', 'birdhouse', 'bookshelf', 'bottle', 'bowl', 'bus', 'cabinet', 'camera', 'can', 'cap', 'car', 'cellular telephone', 'chair', 'clock', 'computer keyboard', 'dishwasher', 'display', 'earphone', 'faucet', 'file', 'guitar', 'helmet', 'jar', 'knife', 'lamp', 'laptop', 'loudspeaker', 'mailbox', 'microphone', 'microwave', 'motorcycle', 'mug', 'piano', 'pillow', 'pistol', 'pot', 'printer', 'remote control', 'rifle', 'rocket', 'skateboard', 'sofa', 'stove', 'table', 'telephone', 'tower', 'train', 'vessel', 'washer']
    
    test_prompts = ['image of a ' + test_prompts[i] for i in range(len(test_prompts))]
    val_prompts = ['image of a ' + val_prompts[i] for i in range(len(val_prompts))]
    
    test_prompts_ = clip.tokenize(test_prompts)
    test_prompt_feats = clip_model.encode_text(test_prompts_)
    test_prompt_feats = test_prompt_feats / test_prompt_feats.norm(dim=-1, keepdim=True)
    
    val_prompts_ = clip.tokenize(val_prompts)
    val_prompt_feats = clip_model.encode_text(val_prompts_)
    val_prompt_feats = val_prompt_feats / val_prompt_feats.norm(dim=-1, keepdim=True)

    # --- Dataloader ---
    train_dataset = ShapeNetRender(partition='train', num_points=args.num_points)
    val_dataset = ShapeNetRender(partition='test', num_points=args.num_points)
    test_dataset = ModelNet40Align(num_points=args.num_points)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, 
                                  sampler=train_sampler, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, num_workers=4, 
                            sampler=val_sampler, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=4, 
                             sampler=test_sampler, pin_memory=True)

    # --- 模型初始化 ---
    if rank == 0:
        summary_writer = SummaryWriter("pre_results/%s/tensorboard" % (args.exp_name))
    else:
        summary_writer = None

    model = CoTeachCLIP(args)
    model = model.to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if args.ckpt is not None:
        if rank == 0:
            io.cprint(f"Loading checkpoint from {args.ckpt}...")
        try:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
            checkpoint = torch.load(args.ckpt, map_location=map_location)
            msg = model.module.load_state_dict(checkpoint, strict=False)
            if rank == 0:
                io.cprint(f"Weights loaded successfully! Msg: {msg}")
        except Exception as e:
            if rank == 0:
                io.cprint(f"[Error] Failed to load checkpoint: {e}")

    # 冻结 Teacher 参数
    for name, param in model.named_parameters():
        if 'image_model' in name or 'text_model' in name:
            param.requires_grad_(False)
            
    val_prompt_feats = val_prompt_feats.to(device)
    test_prompt_feats = test_prompt_feats.to(device)

    # --- 优化器 ---
    optimizer = optim.Lamb(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # --- [SCHEDULER] Standard Cosine Annealing (CoTeachCLIP Style) ---
    # 计算总步数
    steps_per_epoch = len(train_dataloader) // args.accumulation_steps
    total_steps = steps_per_epoch * args.epoch
    
    # 使用标准的 CosineAnnealingLR，没有 Restart
    # LR 将从 args.lr 平滑下降到 1e-6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps, 
        eta_min=1e-6
    )

    n_epochs = args.epoch
    max_val_acc = 0
    max_test_acc = 0

    for epoch in range(n_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        
        # [No Damping Logic here]

        loss_sum = 0
        depth_sum = 0
        image_sum = 0
        text_sum = 0 
        
        iterator = tqdm(train_dataloader) if rank == 0 else train_dataloader
        
        optimizer.zero_grad()

        for i, (image, points, text_tokens, a, e, d) in enumerate(iterator):
            image = image.to(device)
            points = points.to(device)
            text_tokens = text_tokens.to(device)
            a = a.unsqueeze(-1).to(device)
            e = e.unsqueeze(-1).to(device)
            d = d.unsqueeze(-1).to(device)
            
            is_update_step = ((i + 1) % args.accumulation_steps == 0) or ((i + 1) == len(train_dataloader))
            
            context = model.no_sync() if (not is_update_step) else torch.enable_grad()
            
            with context:
                loss, image_loss, text_loss, depth_loss, sim_stats = model(points, image, text_tokens, a, e, d)
                loss = loss / args.accumulation_steps
                loss.backward()

            if is_update_step:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                # [No EMA update]

            with torch.no_grad():
                current_loss_val = loss * args.accumulation_steps
                reduced_loss = reduce_tensor(current_loss_val) / world_size
                reduced_image_loss = reduce_tensor(torch.mean(image_loss)) / world_size
                reduced_text_loss = reduce_tensor(torch.mean(text_loss)) / world_size 
                reduced_depth_loss = reduce_tensor(torch.mean(depth_loss)) / world_size
                
                loss_sum += reduced_loss.item()
                image_sum += reduced_image_loss.item()
                text_sum += reduced_text_loss.item()
                depth_sum += reduced_depth_loss.item()
    
        # --- Validation ---
        model.eval()
        with torch.no_grad():
            correct_num = torch.tensor(0.0).to(device)
            total = torch.tensor(0.0).to(device)
            
            for (points, label) in (tqdm(val_loader) if rank == 0 else val_loader):
                b = points.shape[0]
                points = points.to(device)
                label = label.to(device)
                
                img_feats = model.module.infer(points)
                logits = img_feats @ val_prompt_feats.t()
                logits = logits.reshape(b, args.views, -1)
                logits = torch.sum(logits, dim=1)
                probs = logits.softmax(dim=-1)
                index = torch.max(probs, dim=1).indices
                
                correct_num += torch.sum(torch.eq(index, label)).float()
                total += float(len(label))

            correct_num = reduce_tensor(correct_num)
            total = reduce_tensor(total)
            val_acc = correct_num / total

        # --- Testing ---
        with torch.no_grad():
            correct_num = torch.tensor(0.0).to(device)
            total = torch.tensor(0.0).to(device)
            
            for (points, label) in (tqdm(test_loader) if rank == 0 else test_loader):
                b = points.shape[0]
                points = points.to(device)
                label = label.to(device)
                
                img_feats = model.module.infer(points, True)
                logits = img_feats @ test_prompt_feats.t()
                logits = logits.reshape(b, args.views, -1)
                logits = torch.sum(logits, dim=1)
                probs = logits.softmax(dim=-1)
                index = torch.max(probs, dim=1).indices
                
                correct_num += torch.sum(torch.eq(index, label)).float()
                total += float(len(label))

            correct_num = reduce_tensor(correct_num)
            total = reduce_tensor(total)
            test_acc = correct_num / total

        # [NEW] 如果需要在每个 Batch 监控（可选，仅在 rank 0 打印）
        if rank == 0 and i % 10 == 0:
            print(f"Batch {i} Sim (Img): min={sim_stats['img_min']:.3f}, mean={sim_stats['img_mean']:.3f}, max={sim_stats['img_max']:.3f}")
        # --- Logging ---
        if rank == 0:
            depth_loss_avg = depth_sum / len(train_dataloader)
            image_loss_avg = image_sum / len(train_dataloader)
            text_loss_avg = text_sum / len(train_dataloader)
            mean_loss_avg = loss_sum / len(train_dataloader)
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # [MODIFIED] 获取改进方案中的新参数值用于打印
            with torch.no_grad():
                l_v_t = torch.clamp(model.module.log_var_text, min=-1.0, max=1.0).item()
                l_v_d = torch.clamp(model.module.log_var_depth, min=-1.0, max=1.0).item()
                temp_val = torch.clamp(torch.exp(model.module.log_temperature), min=0.01, max=0.5).item()

            # [MODIFIED] 更新打印信息，包含 log_var_text, log_var_depth 和 temperature
            io.cprint('epoch%d lr: %.6f, total: %.4f, img: %.4f, txt: %.4f, depth: %.4f, '
                      'lv_txt/d: %.3f/%.3f, temp: %.4f, Sim(Img) min/avg/max: %.3f/%.3f/%.3f, val: %.4f, test: %.4f' % 
                      (epoch + 1, current_lr, mean_loss_avg, image_loss_avg, text_loss_avg, depth_loss_avg, 
                       l_v_t, l_v_d, temp_val, 
                       sim_stats['img_min'], sim_stats['img_mean'], sim_stats['img_max'], # 打印项
                       val_acc.item(), test_acc.item()))
            summary_writer.add_scalar('train/sim_img_mean', sim_stats['img_mean'], epoch + 1)
            summary_writer.add_scalar('train/loss', mean_loss_avg, epoch + 1)
            summary_writer.add_scalar('train/lr', current_lr, epoch + 1)
            summary_writer.add_scalar('train/text_loss', text_loss_avg, epoch + 1)
            summary_writer.add_scalar("val/acc", val_acc.item(), epoch + 1)
            summary_writer.add_scalar("test/acc", test_acc.item(), epoch + 1)
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                torch.save(model.module.state_dict(), '%s/%s/best_val.pth' % ('pre_results', args.exp_name))
                io.cprint('save the best val acc at %d' % (epoch + 1))
            if test_acc > max_test_acc:
                max_test_acc = test_acc
                torch.save(model.module.state_dict(), '%s/%s/best_test.pth' % ('pre_results', args.exp_name))
                io.cprint('save the best test acc at %d' % (epoch + 1))

if __name__ == "__main__":
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
    else:
        print("Not running in distributed mode.")
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
        dist.init_process_group(backend='nccl')

    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='moev3+VLM_CoTeachCLIP', metavar='N')
    parser.add_argument('--views', type=int, default=10)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--dim', type=int, default=0, choices=[0, 512])
    parser.add_argument('--model', type=str, default='PointNet', metavar='N')
    parser.add_argument('--batch_size', type=int, default=128, metavar='batch_size')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size')
    parser.add_argument('--epoch', type=int, default=100, metavar='N') 
    parser.add_argument('--use_text_teacher', action='store_true', default=True)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.006, help='Learning rate') 
    
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    if rank == 0:
        _init_('pre_results', args)
        io = IOStream('pre_results' + '/' + args.exp_name + '/run.log')
        io.cprint(str(args))
    else:
        class DummyIO:
            def cprint(self, text): pass
            def close(self): pass
        io = DummyIO()

    train(args, io)
    dist.destroy_process_group()