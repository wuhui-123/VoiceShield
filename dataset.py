import os
import torch
from pathlib import Path
from torch.utils.data import Dataset
from dotenv import load_dotenv

load_dotenv()


class ASVspoof5Dataset(Dataset):
    """
    ASVspoof5 数据集加载器
    
    两种模式：
    - 缓存模式 (use_cache=True)：从预处理好的 .pt 特征文件加载
    - 原始模式 (use_cache=False)：返回文件名和标签，后续实时处理
    """
    
    def __init__(self, split: str, use_cache: bool = True):
        """
        Args:
            split: 'train' 或 'dev'
            use_cache: True=从缓存加载, False=返回原始文件名
        """
        self.split = split
        self.use_cache = use_cache
        
        # ========== 路径配置（从 .env 读取）==========
        self.train_audio_dir = os.getenv('TRAIN_AUDIO_DIR')
        self.dev_audio_dir = os.getenv('DEV_AUDIO_DIR')
        
        self.train_protocol = os.getenv('TRAIN_PROTOCOL')
        self.dev_protocol = os.getenv('DEV_PROTOCOL')
        
        self.train_cache_dir = os.getenv('TRAIN_CACHE_DIR')
        self.dev_cache_dir = os.getenv('DEV_CACHE_DIR')
        # ===========================================
        
        if use_cache:
            self._load_cache()
        else:
            self._load_original()
    
    def _load_cache(self):
        """从缓存加载特征"""
        cache_dir = self.train_cache_dir if self.split == 'train' else self.dev_cache_dir
        
        if not cache_dir or not Path(cache_dir).exists():
            raise FileNotFoundError(f"缓存目录不存在: {cache_dir}")
        
        cache_path = Path(cache_dir)
        labels_path = cache_path / 'labels.pt'
        
        if not labels_path.exists():
            raise FileNotFoundError(f"标签文件不存在: {labels_path}")
        
        self.labels = torch.load(labels_path)
        self.feature_files = sorted([f for f in cache_path.glob("*.pt") if f.name != "labels.pt"])
        
        if len(self.labels) != len(self.feature_files):
            raise RuntimeError(f"缓存不一致: 标签 {len(self.labels)} vs 特征 {len(self.feature_files)}")
        
        num_real = sum(1 for l in self.labels if l == 0)
        num_spoof = sum(1 for l in self.labels if l == 1)
        
        print(f"从缓存加载 {self.split} 集完成:")
        print(f"  总样本数: {len(self.feature_files)}")
        print(f"  真实语音: {num_real} ({num_real/len(self.labels)*100:.1f}%)")
        print(f"  伪造语音: {num_spoof} ({num_spoof/len(self.labels)*100:.1f}%)")
    
    def _load_original(self):
        """从协议文件读取文件名和标签"""
        audio_dir = self.train_audio_dir if self.split == 'train' else self.dev_audio_dir
        protocol_file = self.train_protocol if self.split == 'train' else self.dev_protocol
        
        if not protocol_file or not Path(protocol_file).exists():
            raise FileNotFoundError(f"协议文件不存在: {protocol_file}")
        
        file_names = []
        labels = []
        
        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 9:
                    file_names.append(parts[1])
                    labels.append(0 if parts[8] == 'bonafide' else 1)
        
        self.file_names = file_names
        self.labels = labels
        self.audio_dir = audio_dir
        
        num_real = sum(1 for l in self.labels if l == 0)
        num_spoof = sum(1 for l in self.labels if l == 1)
        
        print(f"从原始数据加载 {self.split} 集完成:")
        print(f"  总样本数: {len(self.file_names)}")
        print(f"  真实语音: {num_real} ({num_real/len(self.labels)*100:.1f}%)")
        print(f"  伪造语音: {num_spoof} ({num_spoof/len(self.labels)*100:.1f}%)")
    
    def __len__(self):
        if self.use_cache:
            return len(self.feature_files)
        return len(self.file_names)
    
    def __getitem__(self, index):
        if self.use_cache:
            # ========== 修改点：适配新的保留时序的 autoencoder ==========
            data = torch.load(self.feature_files[index], weights_only=True)
            
            # 新版 autoencoder 输出：[time, latent_dim]
            compressed = data['compressed']  # [time, latent_dim]
            
            # 可选：保存 time_steps（原始时间步数，用于去padding）
            # 如果预处理时固定了长度，time_steps 可能不需要
            time_steps = data.get('time_steps', compressed.shape[0])
            
            label = self.labels[index]
            
            # 返回格式：[time, latent_dim], label, time_steps
            return compressed, label, time_steps
            # ==========================================================
        else:
            file_name = self.file_names[index]
            label = self.labels[index]
            return file_name, label