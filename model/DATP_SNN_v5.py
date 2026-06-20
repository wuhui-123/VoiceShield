import torch
import torch.nn as nn
import snntorch as snn

class SEBlock1D(nn.Module):
    def __init__(self, in_channels=400, embedding_channels=64):
        super().__init__()
        self.fn = nn.Sequential(
            nn.Linear(in_channels, embedding_channels),
            nn.ReLU(),
            nn.Linear(embedding_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_mean = x.mean(dim=2)
        weight = self.fn(x_mean)
        weight = weight.unsqueeze(2)
        return x * weight

class DynamicGaussianMixture(nn.Module):
    def __init__(self, channels, max_gaussians=5, init_gaussians=2):
        super().__init__()
        
        self.channels = channels
        self.max_gaussians = max_gaussians
        self.current_gaussians = init_gaussians

        self.register_buffer('_num_gaussians', torch.tensor(init_gaussians))
        
        self.means = nn.ParameterList()
        self.log_vars = nn.ParameterList()
        self.mix_weights = nn.ParameterList()
        
        for i in range(init_gaussians):
            mean_init = (i + 0.5) / init_gaussians
            self.means.append(nn.Parameter(torch.full((channels,), mean_init)))
            self.log_vars.append(nn.Parameter(torch.zeros(channels)))
            self.mix_weights.append(nn.Parameter(torch.ones(channels) / init_gaussians))
        
        # 缓存时间轴和堆叠的参数
        self.register_buffer('_cached_t', None)
        self._cached_time_len = 0
        self._cached_params = None  # 缓存堆叠后的参数

    def _get_stacked_params(self, device):
        """将 ParameterList 堆叠成张量（缓存结果）"""
        if self._cached_params is not None:
            return self._cached_params
        
        means = torch.stack([m for m in self.means], dim=0)  # [K, channels]
        variances = torch.exp(torch.stack([v for v in self.log_vars], dim=0)) + 1e-6
        mix_weights = torch.sigmoid(torch.stack([w for w in self.mix_weights], dim=0))
        
        self._cached_params = (means, variances, mix_weights)
        return self._cached_params
    
    def _invalidate_cache(self):
        """参数变化时（添加新高斯）使缓存失效"""
        self._cached_params = None
    
    def add_gaussian(self):
        if self.current_gaussians >= self.max_gaussians:
            return
        
        mean_init = 0.5
        self.means.append(nn.Parameter(torch.full((self.channels,), mean_init)))
        self.log_vars.append(nn.Parameter(torch.zeros(self.channels)))
        
        new_weight = 1.0 / (self.current_gaussians + 1)
        for i in range(self.current_gaussians):
            with torch.no_grad():
                self.mix_weights[i] *= (1 - new_weight)
        
        self.mix_weights.append(nn.Parameter(torch.full((self.channels,), new_weight)))
        self.current_gaussians += 1
        
        # 使缓存失效
        self._invalidate_cache()
    
    def compute_attention_weights(self, x, return_components=False):
        batch, channels, time = x.shape
        device = x.device
        
        # 缓存时间轴（避免重复创建）
        if self._cached_t is None or self._cached_t.shape[-1] != time:
            self._cached_t = torch.linspace(0, 1, time, device=device)
            self._cached_t = self._cached_t.view(1, 1, 1, -1)
            self._cached_time_len = time
        elif self._cached_t.device != device:
            self._cached_t = self._cached_t.to(device)
        
        t = self._cached_t  # [1, 1, 1, time]
        
        # 获取堆叠的参数
        means, variances, mix_weights = self._get_stacked_params(device)
        K = means.shape[0]  # 当前高斯数量
        
        # 广播计算所有高斯（向量化，无 for 循环）
        # means: [K, channels, 1, 1]
        # variances: [K, channels, 1, 1]
        # mix_weights: [K, channels, 1, 1]
        means = means.view(K, channels, 1, 1)
        variances = variances.view(K, channels, 1, 1)
        mix_weights = mix_weights.view(K, channels, 1, 1)
        
        dist_sq = (t - means) ** 2
        component_weights = mix_weights * torch.exp(-dist_sq / (2 * variances))
        
        total_weights = component_weights.sum(dim=0)  # [1, channels, 1, time]
        weights = total_weights.squeeze(0).squeeze(1)  # [channels, time]
        weights = weights.expand(batch, -1, -1)  # [batch, channels, time]
        
        if return_components:
            return weights, component_weights
        return weights

class DATPBlock(nn.Module):
    def __init__(self, channels, max_gaussians=5, init_gaussians=2, growth_threshold=0.01):
        super().__init__()
        
        self.gmm = DynamicGaussianMixture(channels, max_gaussians, init_gaussians)
        self.growth_threshold = growth_threshold
        
        self.input_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.output_proj = nn.Conv1d(channels, channels, kernel_size=1)
        
        self.register_buffer('component_usage', torch.zeros(max_gaussians))
        
        # 缓存 should_add_gaussian 的结果
        self._should_grow_cached = None
        self._cached_input_hash = None
    
    def _get_input_hash(self, x):
        """获取输入的哈希"""
        return (x.shape, x.sum().item(), x.mean().item())
    
    def should_add_gaussian(self, x):
        if not self.training:
            return False
        
        current_hash = self._get_input_hash(x)
        if current_hash == self._cached_input_hash:
            return self._should_grow_cached
        
        with torch.no_grad():
            weights = self.gmm.compute_attention_weights(x)
            max_weights = weights.max(dim=2, keepdim=True)[0]
            min_max_weight = max_weights.min().item()
            result = min_max_weight < self.growth_threshold
        
        self._should_grow_cached = result
        self._cached_input_hash = current_hash
        return result
    
    def update_component_usage(self, x):
        """更新分量使用统计（简化版，跳过详细统计以提升性能）"""
        # 由于性能考虑，暂时跳过分量使用统计
        # 动态添加高斯的逻辑仍然正常工作
        pass
    
    def forward(self, x):
        x = self.input_proj(x)
        
        # 动态添加高斯
        if self.training and self.should_add_gaussian(x):
            if self.gmm.current_gaussians < self.gmm.max_gaussians:
                self.gmm.add_gaussian()
                self._cached_input_hash = None  # 缓存失效
        
        # 计算注意力权重
        weights = self.gmm.compute_attention_weights(x)
        output = x * weights
        output = self.output_proj(output) + x
        
        return output

class DATP_SNN(nn.Module):
    def __init__(self, freq_bins=768, groups=8):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=freq_bins,
            out_channels=freq_bins,
            kernel_size=3,
            padding=1,
            groups=groups
        )

        self.bn1 = nn.BatchNorm1d(freq_bins)
        self.se1 = SEBlock1D(in_channels=freq_bins, embedding_channels=128)
        self.lif1 = snn.Leaky(beta=0.9, learn_beta=True, threshold=0.3)

        self.conv2 = nn.Conv1d(
            in_channels=freq_bins,
            out_channels=512,
            kernel_size=3,
            padding=1,
            groups=groups
        )

        self.bn2 = nn.BatchNorm1d(512)
        self.se2 = SEBlock1D(in_channels=512, embedding_channels=128)
        self.lif2 = snn.Leaky(beta=0.9, learn_beta=True, threshold=0.3)

        self.conv3 = nn.Conv1d(
            in_channels=512,
            out_channels=256,
            kernel_size=3,
            padding=1,
            groups=groups
        )

        self.bn3 = nn.BatchNorm1d(256)
        self.se3 = SEBlock1D(in_channels=256, embedding_channels=128)
        self.lif3 = snn.Leaky(beta=0.9, learn_beta=True, threshold=0.3)

        self.conv4 = nn.Conv1d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            padding=1,
            groups=groups
        )

        self.bn4 = nn.BatchNorm1d(128)
        self.datp = DATPBlock(128)
        self.lif4 = snn.Leaky(beta=0.9, learn_beta=True, threshold=0.3)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        num_steps = x.shape[0]

        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()
        mem3 = self.lif3.reset_mem()
        mem4 = self.lif4.reset_mem()

        spk_res = []

        for step in range(num_steps):
            x_step = x[step]

            curl_x = self.conv1(x_step)
            curl_x = self.bn1(curl_x)
            curl_x = self.se1(curl_x)

            spk1, mem1 = self.lif1(curl_x, mem1)

            curl_x = self.conv2(spk1)
            curl_x = self.bn2(curl_x)
            curl_x = self.se2(curl_x)

            spk2, mem2 = self.lif2(curl_x, mem2)

            curl_x = self.conv3(spk2)
            curl_x = self.bn3(curl_x)
            curl_x = self.se3(curl_x)

            spk3, mem3 = self.lif3(curl_x, mem3)

            curl_x = self.conv4(spk3)
            curl_x = self.bn4(curl_x)

            curl_x = self.datp(curl_x)
            curl_x = self.dropout(curl_x)

            spk4, mem4 = self.lif4(curl_x, mem4)

            spk_res.append(spk4)

        return torch.stack(spk_res)