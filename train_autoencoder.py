# train_autoencoder.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from dotenv import load_dotenv

os.environ['HF_ENDPOINT'] = os.getenv('HF_ENDPOINT', 'https://hf-mirror.com')
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from dataset import ASVspoof5Dataset
from models.autoencoder import Wav2Vec2AutoEncoder


def collate_fn(batch):
    """返回文件名和标签"""
    file_names, labels = zip(*batch)
    return list(file_names), torch.tensor(labels)


def load_batch_audio(file_names, audio_dirs, sr, target_len):
    """
    批量加载音频文件
    
    Args:
        file_names: 文件名列表
        audio_dirs: {'train': path, 'dev': path}
        sr: 采样率
        target_len: 目标长度（采样点数）
    
    Returns:
        waveforms: list of numpy arrays
    """
    waveforms = []
    for file_name in file_names:
        # 根据文件名前缀判断目录
        prefix = file_name[0].upper()
        if prefix == 'T':
            audio_dir = audio_dirs['train']
        elif prefix == 'D':
            audio_dir = audio_dirs['dev']
        else:
            # 尝试两个目录
            for key in ['train', 'dev']:
                if Path(audio_dirs[key], f"{file_name}.flac").exists():
                    audio_dir = audio_dirs[key]
                    break
            else:
                raise FileNotFoundError(f"找不到音频文件: {file_name}.flac")
        
        audio_path = Path(audio_dir) / f"{file_name}.flac"
        waveform, _ = librosa.load(audio_path, sr=sr)
        
        # 统一长度
        if len(waveform) < target_len:
            waveform = np.pad(waveform, (0, target_len - len(waveform)))
        else:
            waveform = waveform[:target_len]
        
        waveforms.append(waveform)
    
    return waveforms


def get_feature_mask(input_attention_mask, feature_length):
    batch_size = input_attention_mask.shape[0]
    max_original = input_attention_mask.shape[1]
    
    # 计算每个样本的原始有效长度（采样点数量）
    original_lengths = input_attention_mask.sum(dim=1)  # [batch]
    
    # 线性映射到特征帧数
    # original_length / max_original = feature_length_i / feature_length
    feature_lengths = torch.ceil(
        original_lengths.float() * feature_length / max_original
    ).long()
    feature_lengths = torch.clamp(feature_lengths, min=1, max=feature_length)
    
    # 生成 mask
    mask = torch.zeros(batch_size, feature_length, 
                       device=input_attention_mask.device,
                       dtype=torch.float32)
    
    for i, length in enumerate(feature_lengths):
        mask[i, :length] = 1.0
    
    return mask


def train_autoencoder(
    sr: int,
    audio_length: float,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    latent_dim: int,
    seed: int,
    model_name: str,
    max_samples: int,
    autoencoder_path: str,
    num_workers: int = 4,
    use_amp: bool = True
):
    """
    训练自编码器（批量提取 Wav2Vec2 特征，GPU 并行加速）
    
    Args:
        sr: 采样率
        audio_length: 音频长度（秒）
        batch_size: 批大小
        epochs: 训练轮数
        lr: 学习率
        weight_decay: 权重衰减
        patience: 早停耐心值
        latent_dim: 压缩维度
        seed: 随机种子
        model_name: Wav2Vec2 模型名
        max_samples: 最大训练样本数（0=全部）
        autoencoder_path: 模型保存路径
        num_workers: DataLoader 工作进程数
        use_amp: 是否使用混合精度训练
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 混合精度只在 CUDA 下有效
    use_amp = use_amp and device.type == 'cuda'
    print(f"混合精度: {'开启' if use_amp else '关闭'}")
    
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    print("\n" + "=" * 60)
    print("加载数据集...")
    train_dataset = ASVspoof5Dataset(split='train', use_cache=False)
    dev_dataset = ASVspoof5Dataset(split='dev', use_cache=False)
    
    audio_dirs = {
        'train': train_dataset.train_audio_dir,
        'dev': dev_dataset.dev_audio_dir
    }
    
    full_dataset = ConcatDataset([train_dataset, dev_dataset])
    total_size = len(full_dataset)
    print(f"  总样本数: {total_size}")
    
    if max_samples > 0 and max_samples < total_size:
        indices = torch.randperm(total_size)[:max_samples].tolist()
        full_dataset = Subset(full_dataset, indices)
        print(f"  随机采样 {max_samples} 条数据用于训练")
    
    # 创建 DataLoader
    loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    print(f"  DataLoader: batch_size={batch_size}, workers={num_workers}")
    print(f"  每 epoch 迭代次数: {len(loader)}")
    
    print("\n" + "=" * 60)
    print("加载模型...")
    print(f"  Wav2Vec2: {model_name}")
    print(f"  压缩维度: 768 -> {latent_dim}")
    
    # Wav2Vec2 processor（用于批量处理）
    wav2vec2_processor = Wav2Vec2Processor.from_pretrained(model_name)
    
    # Wav2Vec2 encoder（冻结）
    wav2vec2_model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    wav2vec2_model.eval()
    for param in wav2vec2_model.parameters():
        param.requires_grad = False
    print(f"  Wav2Vec2 加载完成（参数已冻结）")
    
    # 自编码器
    autoencoder = Wav2Vec2AutoEncoder(
        input_dim=768,
        latent_dim=latent_dim
    ).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in autoencoder.parameters())
    trainable_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
    print(f"  自编码器参数量: {total_params:,} (可训练: {trainable_params:,})")
    
    # 优化器和调度器
    optimizer = AdamW(
        autoencoder.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    
    # 损失函数
    criterion = nn.MSELoss()
    
    # 混合精度 scaler
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    target_len = int(sr * audio_length)
    best_loss = float('inf')
    patience_counter = 0
    
    # 创建保存目录
    save_dir = os.path.dirname(autoencoder_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    print(f"  音频长度: {audio_length}s -> {target_len} 采样点 @ {sr}Hz")
    print(f"  学习率: {lr} (min: {lr * 0.01})")
    print(f"  权重衰减: {weight_decay}")
    print(f"  早停耐心值: {patience}")
    print(f"  模型保存路径: {autoencoder_path}")
    print(f"  混合精度: {'AMP' if use_amp else 'FP32'}")
    print("=" * 60)
    
    for epoch in range(epochs):
        autoencoder.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for file_names, labels in pbar:
            # ---------- 批量加载音频 ----------
            waveforms = load_batch_audio(
                file_names=file_names,
                audio_dirs=audio_dirs,
                sr=sr,
                target_len=target_len
            )
            
            inputs = wav2vec2_processor(
                waveforms,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True
            ).to(device)
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    with torch.no_grad():
                        features = wav2vec2_model(**inputs).last_hidden_state
                        # features: [batch, time_frames, 768]
                    
                    compressed, reconstructed = autoencoder(features)
                    # reconstructed: [batch, time_frames, 768]
                    
                    # 生成特征级别的 mask（根据原始音频有效长度按比例映射）
                    feature_mask = get_feature_mask(
                        inputs.attention_mask,   # [batch, original_time]
                        features.shape[1]        # time_frames
                    )
                    # feature_mask: [batch, time_frames]
                    
                    # 扩展 mask 到特征维度
                    mask = feature_mask.unsqueeze(-1)  # [batch, time_frames, 1]
                    
                    # 计算加权 MSE 损失（忽略 padding 帧）
                    loss = (criterion(reconstructed, features) * mask).sum() / mask.sum()
                
                # 反向传播
                optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                with torch.no_grad():
                    features = wav2vec2_model(**inputs).last_hidden_state
                
                compressed, reconstructed = autoencoder(features)
                
                feature_mask = get_feature_mask(
                    inputs.attention_mask,
                    features.shape[1]
                )
                mask = feature_mask.unsqueeze(-1)
                loss = (criterion(reconstructed, features) * mask).sum() / mask.sum()
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
                optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{total_loss/num_batches:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        # 更新学习率
        scheduler.step()
        avg_loss = total_loss / num_batches
        
        print(f"Epoch {epoch+1}/{epochs} - 平均损失: {avg_loss:.6f}, 学习率: {scheduler.get_last_lr()[0]:.2e}")
        
        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # 保存完整 checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': autoencoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'latent_dim': latent_dim,
                'config': {
                    'sr': sr,
                    'audio_length': audio_length,
                    'latent_dim': latent_dim,
                    'input_dim': 768,
                }
            }
            torch.save(checkpoint, autoencoder_path)
            print(f"  ✅ 保存最佳模型 (loss: {best_loss:.6f})")
        else:
            patience_counter += 1
            print(f"  验证损失未改善 ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"  🛑 早停触发！")
                break
        
        print("-" * 60)
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"  最佳损失: {best_loss:.6f}")
    print(f"  模型保存至: {autoencoder_path}")
    print("=" * 60)
    
    return autoencoder, best_loss


if __name__ == "__main__":
    load_dotenv()
    
    # 从环境变量读取配置
    config = {
        'sr': int(os.getenv('SR', 16000)),
        'audio_length': float(os.getenv('AUDIO_LENGTH', 6.0)),
        'batch_size': int(os.getenv('AE_BATCH_SIZE', 128)),
        'epochs': int(os.getenv('AE_NUM_EPOCHS', 100)),
        'lr': float(os.getenv('AE_LEARNING_RATE', 1e-4)),
        'weight_decay': float(os.getenv('AE_WEIGHT_DECAY', 1e-5)),
        'patience': int(os.getenv('AE_PATIENCE', 15)),
        'latent_dim': int(os.getenv('AE_LATENT_DIM', 128)),
        'seed': int(os.getenv('SEED', 42)),
        'model_name': os.getenv('WAV2VEC2_MODEL', 'facebook/wav2vec2-base'),
        'max_samples': int(os.getenv('AE_MAX_SAMPLES', 0)),
        'autoencoder_path': os.getenv('AE_MODEL_PATH', 'checkpoints/autoencoder_best.pt'),
        'num_workers': int(os.getenv('AE_NUM_WORKERS', 4)),
        'use_amp': os.getenv('AE_USE_AMP', 'true').lower() == 'true'
    }
    
    print("=" * 60)
    print("自编码器训练配置")
    print("=" * 60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # 开始训练
    model, best_loss = train_autoencoder(**config)
