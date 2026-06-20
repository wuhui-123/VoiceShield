import os
import torch
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from dotenv import load_dotenv

os.environ['HF_ENDPOINT'] = os.getenv('HF_ENDPOINT', 'https://hf-mirror.com')
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from dataset import ASVspoof5Dataset
from models.autoencoder import Wav2Vec2AutoEncoder


def collate_for_preprocessing(batch):
    """预处理专用 collate：只返回文件名和标签"""
    file_names, labels = zip(*batch)
    return list(file_names), torch.tensor(labels)


def extract_and_compress_batch(
    file_names: list,
    audio_dir: str,
    sr: int,
    target_len: int,
    wav2vec2_model: Wav2Vec2Model,
    wav2vec2_processor: Wav2Vec2Processor,
    autoencoder: Wav2Vec2AutoEncoder,
    device: torch.device
):
    """
    批量提取 Wav2Vec2 特征并压缩
    
    Returns:
        compressed_features: [batch, time, latent_dim] - 保留时序
        time_steps: list - 每个样本原始时间步数（用于后续恢复）
    """
    # 1. 批量加载音频
    waveforms = []
    for file_name in file_names:
        audio_path = Path(audio_dir) / f"{file_name}.flac"
        waveform, _ = librosa.load(audio_path, sr=sr)
        
        # 统一长度到 target_len
        if len(waveform) < target_len:
            waveform = np.pad(waveform, (0, target_len - len(waveform)))
        else:
            waveform = waveform[:target_len]
        
        waveforms.append(waveform)
    
    # 2. 批量提取 Wav2Vec2 特征
    waveforms = np.stack(waveforms)
    inputs = wav2vec2_processor(
        waveforms.tolist(),
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    ).to(device)
    
    with torch.no_grad():
        wav2vec2_features = wav2vec2_model(**inputs).last_hidden_state
        # [batch, time, 768]
    
    # 记录每个样本原始时间步（padding前）
    time_steps = inputs.attention_mask.sum(dim=1).tolist()
    
    # 3. 自编码器压缩
    compressed_features = autoencoder.encode(wav2vec2_features)
    # [batch, time, latent_dim]
    
    return compressed_features, time_steps


def preprocess_dataset(
    split: str,
    autoencoder_path: str,
    batch_size: int = 32,
    audio_length: int = 6,
    sr: int = 16000,
    latent_dim: int = 128,
    model_name: str = "facebook/wav2vec2-base"
):
    """
    预处理数据集：提取特征 → 压缩 → 保存为多个独立 .pt 文件
    
    Args:
        split: 'train' 或 'dev'
        autoencoder_path: 自编码器权重路径
        batch_size: 批处理大小
        audio_length: 音频长度（秒）
        sr: 采样率
        latent_dim: 压缩维度
        model_name: Wav2Vec2 模型名
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # ============ 1. 加载数据集 ============
    print(f"加载 {split} 数据集...")
    dataset = ASVspoof5Dataset(split=split, use_cache=False)
    
    # 获取音频目录
    audio_dir = dataset.train_audio_dir if split == 'train' else dataset.dev_audio_dir
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 主进程处理，避免多进程GPU冲突
        collate_fn=collate_for_preprocessing,
        pin_memory=True
    )
    
    # ============ 2. 加载模型 ============
    print("加载模型...")
    
    # Wav2Vec2 模型和处理器
    wav2vec2_processor = Wav2Vec2Processor.from_pretrained(model_name)
    wav2vec2_model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    wav2vec2_model.eval()
    for param in wav2vec2_model.parameters():
        param.requires_grad = False
    
    # 自编码器
    autoencoder = Wav2Vec2AutoEncoder(
        input_dim=768,
        latent_dim=latent_dim
    ).to(device)
    
    if os.path.exists(autoencoder_path):
        checkpoint = torch.load(autoencoder_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            autoencoder.load_state_dict(checkpoint['model_state_dict'])
        else:
            autoencoder.load_state_dict(checkpoint)
        print(f"✅ 加载自编码器: {autoencoder_path}")
        print(f"   latent_dim: {latent_dim}")
    else:
        raise FileNotFoundError(f"自编码器模型不存在: {autoencoder_path}")
    
    autoencoder.eval()
    
    # ============ 3. 准备缓存目录 ============
    cache_dir = dataset.train_cache_dir if split == 'train' else dataset.dev_cache_dir
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    target_len = int(sr * audio_length)
    
    # ============ 4. 批量处理并逐个保存 ============
    all_labels = []
    sample_count = 0
    
    print(f"开始预处理 {split} 集...")
    with torch.no_grad():
        for file_names, labels in tqdm(loader, desc=f"预处理 {split}"):
            # 批量提取并压缩
            compressed_batch, time_steps = extract_and_compress_batch(
                file_names=file_names,
                audio_dir=audio_dir,
                sr=sr,
                target_len=target_len,
                wav2vec2_model=wav2vec2_model,
                wav2vec2_processor=wav2vec2_processor,
                autoencoder=autoencoder,
                device=device
            )
            
            # 逐个保存为独立 .pt 文件
            for i, file_name in enumerate(file_names):
                compressed = compressed_batch[i].cpu()  # [time, latent_dim]
                time_step = int(time_steps[i])
                
                save_data = {
                    'compressed': compressed,      # [time, latent_dim] - 保留时序
                    'time_steps': time_step        # 原始时间步数（去padding用）
                }
                
                save_path = cache_path / f"{file_name}.pt"
                torch.save(save_data, save_path)
                
                all_labels.append(labels[i].item())
                sample_count += 1
    
    # ============ 5. 保存标签文件 ============
    labels_path = cache_path / 'labels.pt'
    torch.save(all_labels, labels_path)
    
    # ============ 6. 统计信息 ============
    print(f"\n✅ 预处理完成！")
    print(f"   保存路径: {cache_dir}")
    print(f"   总样本数: {sample_count}")
    print(f"   标签文件: {labels_path}")
    
    # 随机检查一个样本的形状
    sample_file = list(cache_path.glob("*.pt"))[0]
    sample_data = torch.load(sample_file, weights_only=True)
    print(f"   特征形状示例: {sample_data[0].shape}")  # [time, latent_dim]
    print(f"   时间步示例: {sample_data['time_steps']}")
    
    # 计算总存储大小
    total_size = sum(f.stat().st_size for f in cache_path.glob("*.pt"))
    total_size_mb = total_size / (1024 * 1024)
    print(f"   总存储大小: {total_size_mb:.1f} MB")
    
    # 类别分布
    labels_array = torch.tensor(all_labels)
    unique, counts = torch.unique(labels_array, return_counts=True)
    print(f"   类别分布:")
    for label, count in zip(unique.tolist(), counts.tolist()):
        print(f"     类别 {label}: {count} ({100*count/len(all_labels):.1f}%)")
    
    # 计算压缩比（相对768维原始特征）
    # 假设原始大小：N × max_time × 768 × 4 bytes
    first_sample = torch.load(list(cache_path.glob("*.pt"))[0], weights_only=True)
    time_dim = first_sample['compressed'].shape[0]
    latent_dim_actual = first_sample['compressed'].shape[1]
    
    original_estimated = sample_count * time_dim * 768 * 4  # bytes
    compression_ratio = original_estimated / total_size
    print(f"   估计压缩比: {compression_ratio:.1f}x (相对768维)")
    
    return cache_path


if __name__ == "__main__":
    load_dotenv()
    
    # 从环境变量读取配置
    sr = int(os.getenv('SR', 16000))
    audio_length = int(os.getenv('AUDIO_LENGTH', 6))
    latent_dim = int(os.getenv('AE_LATENT_DIM', 128))
    autoencoder_path = os.getenv('AE_MODEL_PATH', 'checkpoints/autoencoder_best.pt')
    model_name = os.getenv('WAV2VEC2_MODEL', 'facebook/wav2vec2-base')
    
    print("=" * 60)
    print("配置信息:")
    print(f"  采样率: {sr}")
    print(f"  音频长度: {audio_length}s")
    print(f"  压缩维度: {latent_dim}")
    print(f"  自编码器: {autoencoder_path}")
    print(f"  Wav2Vec2: {model_name}")
    print("=" * 60)
    
    # 预处理训练集和开发集
    for split_name in ['train', 'dev']:
        print(f"\n{'='*60}")
        print(f"开始预处理 {split_name} 集")
        print(f"{'='*60}")
        preprocess_dataset(
            split=split_name,
            autoencoder_path=autoencoder_path,
            batch_size=32,  # 根据GPU显存调整
            audio_length=audio_length,
            sr=sr,
            latent_dim=latent_dim,
            model_name=model_name
        )
    
    print("\n🎉 全部预处理完成！")