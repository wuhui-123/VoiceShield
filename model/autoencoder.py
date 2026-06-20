import torch.nn as nn

class Wav2Vec2AutoEncoder(nn.Module):
    """
    自编码器：将Wav2Vec2输出特征压缩，保留时序结构
    
    输入形状：[batch, time, 768]
    输出形状：
        compressed:   [batch, time, latent_dim]  - 压缩后的特征（保留时序）
        decompressed: [batch, time, 768]          - 重建后的特征
    
    使用场景：
        1. 训练阶段：输入特征，同时得到压缩特征和重建特征，用重建损失训练
        2. 预处理阶段：用 encode() 提取压缩特征并保存到磁盘
        3. 推理阶段：直接使用压缩特征训练分类器，或用 decode() 恢复原始维度
    
    压缩比：768 -> 128 = 6倍，时序长度保持不变
    """
    
    def __init__(self, input_dim=768, latent_dim=128, dropout=0.2):
        """
        Args:
            input_dim: 输入特征维度，Wav2Vec2-base 为 768
            latent_dim: 压缩后的维度，默认 128（768/6）
            dropout: Dropout 比率
        """
        super().__init__()
        
        # ============ 编码器 ============
        # [batch, 768, time] -> [batch, latent_dim, time]
        self.encoder = nn.Sequential(
            # 768 -> 512
            nn.Conv1d(input_dim, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # 512 -> 256
            nn.Conv1d(512, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # 256 -> latent_dim
            nn.Conv1d(256, latent_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
        )
        
        # ============ 解码器 ============
        # [batch, latent_dim, time] -> [batch, 768, time]
        self.decoder = nn.Sequential(
            # latent_dim -> 256
            nn.Conv1d(latent_dim, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # 256 -> 512
            nn.Conv1d(256, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # 512 -> input_dim
            nn.Conv1d(512, input_dim, kernel_size=5, padding=2),
            nn.Tanh(),  # 匹配 Wav2Vec2 输出分布 [-1, 1]
        )
    
    def forward(self, x):
        """
        完整的前向传播，用于训练自编码器
        
        Args:
            x: 输入特征 [batch, time, input_dim]
            
        Returns:
            compressed: 压缩特征 [batch, time, latent_dim]
            decompressed: 重建特征 [batch, time, input_dim]
        """
        # [batch, time, 768] -> [batch, 768, time]
        x = x.transpose(1, 2)
        
        # 编码：[batch, 768, time] -> [batch, latent_dim, time]
        compressed = self.encoder(x)
        
        # 解码：[batch, latent_dim, time] -> [batch, 768, time]
        decompressed = self.decoder(compressed)
        
        # 转回原始格式
        # [batch, 768, time] -> [batch, time, 768]
        decompressed = decompressed.transpose(1, 2)
        
        # [batch, latent_dim, time] -> [batch, time, latent_dim]
        compressed = compressed.transpose(1, 2)
        
        return compressed, decompressed
    
    def encode(self, x):
        """
        仅编码，用于预处理阶段提取压缩特征
        
        Args:
            x: 输入特征 [batch, time, input_dim]
            
        Returns:
            compressed: 压缩特征 [batch, time, latent_dim]
        """
        # [batch, time, 768] -> [batch, 768, time]
        x = x.transpose(1, 2)
        compressed = self.encoder(x)
        return compressed.transpose(1, 2)  # [batch, time, latent_dim]
    
    def decode(self, z):
        """
        仅解码，用于将压缩特征恢复为原始维度
        
        Args:
            z: 压缩特征 [batch, time, latent_dim]
            
        Returns:
            decompressed: 恢复的时序特征 [batch, time, input_dim]
        """
        # [batch, time, latent_dim] -> [batch, latent_dim, time]
        z = z.transpose(1, 2)
        decompressed = self.decoder(z)
        return decompressed.transpose(1, 2)  # [batch, time, 768]