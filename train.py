import os

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from tqdm import tqdm
from dotenv import load_dotenv

from dataset import ASVspoof5Dataset
from models.DATP_SNN_v5 import DATP_SNN
from models.autoencoder import Wav2Vec2AutoEncoder
from utils.calculations import compute_logits, compute_eer, compute_min_dcf, FocalLoss

def train_snn(
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    clip_grad_norm: float,
    seed: int,
    device: torch.device,
    autoencoder_path: str,
    num_workers: int = 4,
    use_focal_loss: bool = False,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0
):
    """训练 SNN 模型"""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. 加载数据集（缓存模式）
    print("加载训练集...")
    train_dataset = ASVspoof5Dataset(split='train', use_cache=True)

    print("加载验证集...")
    val_dataset = ASVspoof5Dataset(split='dev', use_cache=True)
    
    if not train_dataset.use_cache or not val_dataset.use_cache:
        print("⚠️ 警告: 未使用缓存模式，请先运行 preprocess.py 预处理数据")
        return
    
    # 2. 加载自编码器（用于解码）
    print("加载自编码器...")
    latent_dim = int(os.getenv('AE_LATENT_DIM', 64))
    autoencoder = Wav2Vec2AutoEncoder(input_dim=768, latent_dim=latent_dim).to(device)
    
    if os.path.exists(autoencoder_path):
        autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))
        print(f"✅ 加载自编码器: {autoencoder_path}")
    else:
        raise FileNotFoundError(f"自编码器模型不存在: {autoencoder_path}")
    
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False
    
    # 3. 创建 SNN 模型（直接使用，freq_bins=768）
    print("创建 SNN 模型...")
    model = DATP_SNN(freq_bins=768, groups=8).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 损失函数
    if use_focal_loss:
        num_real = sum(1 for l in train_dataset.labels if l == 0)
        num_spoof = sum(1 for l in train_dataset.labels if l == 1)

        focal_alpha = num_real / (num_real + num_spoof)

        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        print(f"使用 Focal Loss (alpha={focal_alpha:.4f}, gamma={focal_gamma})")
    else:
        num_real = sum(1 for l in train_dataset.labels if l == 0)
        num_spoof = sum(1 for l in train_dataset.labels if l == 1)
        pos_weight = torch.tensor([num_real / num_spoof]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"使用加权 BCE Loss (pos_weight={pos_weight.item():.4f})")
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 训练
    best_f1 = 0
    best_eer = 1.0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_logits = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for compressed, labels, time_steps in pbar:
            compressed = compressed.to(device)
            labels = labels.to(device).float()
            
            # 使用自编码器解码：将压缩特征恢复为时序特征
            # compressed: [batch, 64]
            # time_steps: [batch] 每个样本的原始时间步长
            batch_features = []
            for i in range(compressed.shape[0]):
                z = compressed[i:i+1]  # [1, 64]
                t = int(time_steps[i].item())
                # decode: [1, 64] -> [1, t, 768]
                features = autoencoder.decode(z, t)  # [1, t, 768]
                batch_features.append(features.squeeze(0))  # [t, 768]
            
            # 对齐时间步长（填充到相同长度）
            max_time = max(f.shape[0] for f in batch_features)
            padded_features = []
            for f in batch_features:
                if f.shape[0] < max_time:
                    pad = torch.zeros(max_time - f.shape[0], f.shape[1], device=device)
                    f = torch.cat([f, pad], dim=0)
                padded_features.append(f)
            
            features = torch.stack(padded_features)  # [batch, max_time, 768]
            
            # 调整形状为 SNN 输入 [steps, batch, 768, time]
            features = features.permute(1, 0, 2)  # [time, batch, 768]
            features = features.unsqueeze(3)       # [time, batch, 768, 1]
            
            optimizer.zero_grad()
            
            if scaler:
                with autocast('cuda'):
                    output = model(features)
                    logits = compute_logits(output)
                    loss = criterion(logits, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(features)
                logits = compute_logits(output)
                loss = criterion(logits, labels)
                loss.backward()
                clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
            
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.detach().cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds)
        train_auc = roc_auc_score(all_labels, all_logits)
        
        # 验证
        model.eval()
        val_labels = []
        val_logits = []
        
        with torch.no_grad():
            for compressed, labels, time_steps in tqdm(val_loader, desc="验证"):
                compressed = compressed.to(device)
                labels = labels.to(device).float()
                
                # 解码
                batch_features = []
                for i in range(compressed.shape[0]):
                    z = compressed[i:i+1]
                    t = int(time_steps[i].item())
                    features = autoencoder.decode(z, t)
                    batch_features.append(features.squeeze(0))
                
                max_time = max(f.shape[0] for f in batch_features)
                padded_features = []
                for f in batch_features:
                    if f.shape[0] < max_time:
                        pad = torch.zeros(max_time - f.shape[0], f.shape[1], device=device)
                        f = torch.cat([f, pad], dim=0)
                    padded_features.append(f)
                
                features = torch.stack(padded_features)
                features = features.permute(1, 0, 2).unsqueeze(3)
                
                output = model(features)
                logits = compute_logits(output)
                
                val_labels.extend(labels.cpu().numpy())
                val_logits.extend(logits.cpu().numpy())

        val_logits = np.array(val_logits)
        val_probs = 1 / (1 + np.exp(-val_logits))
        val_preds = (val_probs > 0.5).astype(int)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_logits)
        eer, eer_threshold = compute_eer(np.array(val_labels), val_logits)
        min_dcf = compute_min_dcf(np.array(val_labels), val_logits)
        tn, fp, fn, tp = confusion_matrix(val_labels, val_preds, labels=[0, 1]).ravel()
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  训练 Loss: {total_loss/len(train_loader):.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
        print(f"  验证 Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}, EER: {eer:.4f}, minDCF: {min_dcf:.4f}")
        print(f"  混淆矩阵 TP:{tp}, TN:{tn}, FP:{fp}, FN:{fn}")
        
        scheduler.step(eer)
        
        if best_eer > eer:
            best_f1 = val_f1
            best_eer = eer
            torch.save(model.state_dict(), "best_snn_model.pth")
            print(f"  ✅ 保存最佳模型 (F1: {best_f1:.4f}, EER: {best_eer:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"早停触发，训练结束")
                break
    
    print(f"\n训练完成！最佳 F1: {best_f1:.4f}, 最佳 EER: {best_eer:.4f}")

if __name__ == "__main__":
    load_dotenv()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    train_snn(
        batch_size=int(os.getenv('SNN_BATCH_SIZE')),
        epochs=int(os.getenv('SNN_NUM_EPOCHS')),
        lr=float(os.getenv('SNN_LEARNING_RATE')),
        weight_decay=float(os.getenv('SNN_WEIGHT_DECAY')),
        clip_grad_norm=float(os.getenv('SNN_CLIP_GRAD_NORM')),
        seed=int(os.getenv('SEED')),
        device=device,
        autoencoder_path=os.getenv('AE_MODEL_PATH', 'best_autoencoder.pth'),  # 新增
        num_workers=int(os.getenv('NUM_WORKERS')),
        use_focal_loss=os.getenv('USE_FOCAL_LOSS').lower() == 'true',
        focal_alpha=float(os.getenv('FOCAL_ALPHA')),
        focal_gamma=float(os.getenv('FOCAL_GAMMA'))
    )