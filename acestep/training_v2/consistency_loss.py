import torch
import torch.nn.functional as F

def compute_latent_frequency_loss(
    pred_x0: torch.Tensor, 
    target_x0: torch.Tensor, 
    fft_weight: float = 1.0, 
    diff_weight: float = 1.0
) -> dict:
    """
    计算频域感知的一致性损失 (Latent Frequency-Aware Loss)。
    """
    # 1. 时域 MSE 损失
    loss_time_mse = F.mse_loss(pred_x0, target_x0)
    
    # 2. 1D-FFT 振幅 L1 损失 
    # [CRITICAL FIX] 必须使用 norm="forward" 或 "ortho"，防止振幅随时间帧数 T 线性爆炸
    pred_fft = torch.fft.rfft(pred_x0, dim=1, norm="forward")
    target_fft = torch.fft.rfft(target_x0, dim=1, norm="forward")
    
    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)
    
    loss_freq_l1 = F.l1_loss(pred_mag, target_mag)
    
    # 3. 时域一阶差分损失 (L1 鼓励差分稀疏，保留瞬态)
    pred_diff = pred_x0[:, 1:, :] - pred_x0[:, :-1, :]
    target_diff = target_x0[:, 1:, :] - target_x0[:, :-1, :]
    loss_diff = F.l1_loss(pred_diff, target_diff)
    
    # 4. 计算总损失
    loss_total = loss_time_mse + fft_weight * loss_freq_l1 + diff_weight * loss_diff
    
    return {
        "loss_total": loss_total,
        "loss_time_mse": loss_time_mse,
        "loss_freq_l1": loss_freq_l1,
        "loss_diff": loss_diff
    }