"""
Neural network models for Stein shift detection.
"""

import torch
import torch.nn as nn


class ClassifierNet(nn.Module):
    """AlexNet-like architecture adapted to 1-channel MNIST (small)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), # 64x64
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),                # 32x32 (AvgPool for higher-order diff)
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),                # 16x16 (AvgPool for higher-order diff)
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))                           # -> 1x1 (MPS-compatible, works for any input size)
        )
        self.classifier = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(256*1*1, 512),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B,1,64,64)
        feats = self.features(x)
        feats = torch.flatten(feats, 1)
        logits = self.classifier(feats)
        return logits


class SmallScoreNet(nn.Module):
    """
    Very small convnet returning per-pixel score estimate s(x) of shape (B,1,64,64).
    Intended for demo / quick experiments on MNIST only.
    For production / best results use a U-Net / diffusion architecture and multi-noise DSM.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x):
        # returns shape (B,1,64,64) approximating gradient of log-density wrt pixels
        return self.net(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel, padding=padding),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )
    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(in_ch, out_ch)
    def forward(self, x):
        return self.block(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.block = ConvBlock(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class UNetScore(nn.Module):
    """
    Small U-Net conditioned on log-sigma via embedding injection (standard approach).
    Output is a per-pixel vector of same shape as input (1 channel), representing score.
    For images, we predict score = grad_x log p_sigma(x) (i.e., score at sigma).
    
    Supports both grayscale (in_channels=1) and RGB (in_channels=3) images.
    Sigma conditioning is done via learned embeddings added to feature maps, not channel concatenation.
    """
    def __init__(self, base_ch=128, in_channels=1):
        """
        Args:
            base_ch: Base number of channels in the network (default 128 for better capacity)
            in_channels: Number of input image channels (1 for grayscale, 3 for RGB)
        """
        super().__init__()
        self.in_channels = in_channels
        self.in_block = ConvBlock(in_channels, base_ch)  # Only image channels (no sigma channel)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.mid = ConvBlock(base_ch*4, base_ch*4)
        self.up2 = Up(base_ch*4 + base_ch*2, base_ch*2)
        self.up1 = Up(base_ch*2 + base_ch, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

        # small embedding MLP for sigma scalar -> broadcast channel
        self.sigma_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, base_ch)  # we'll broadcast to add into feature maps
        )

    def forward(self, x, sigma):
        """
        x: (B, in_channels, H, W) - grayscale (1) or RGB (3)
        sigma: (B,) or scalar (std dev)
        
        Sigma conditioning via embedding injection (standard approach), not channel concatenation.
        """
        B = x.shape[0]
        if sigma.dim() == 0:
            sigma = sigma.view(1).expand(B)
        sigma = sigma.view(B, 1)
        
        # Embedding injection: convert sigma to learned embedding
        s_emb = self.sigma_mlp(sigma)  # (B, base_ch)
        
        # Process image (no sigma channel concatenation)
        h0 = self.in_block(x)  # (B, base_ch, H, W)
        
        # Add sigma embedding to features (standard conditioning approach)
        h0 = h0 + s_emb.view(B, -1, 1, 1)
        h1 = self.down1(h0)      # (B, base_ch*2, H/2, W/2)
        h2 = self.down2(h1)      # (B, base_ch*4, H/4, W/4)
        m = self.mid(h2)
        u2 = self.up2(m, h1)
        u1 = self.up1(u2, h0)
        out = self.out_conv(u1)  # (B,1,H,W)
        # we predict the score for noisy image x: grad_x log p_sigma(x)
        return out


class ClassifierNet32x32(nn.Module):
    """AlexNet-like architecture adapted to 1-channel 32x32 input (e.g., MNIST/Fashion-MNIST resized)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), # 32x32
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),                # 16x16 (AvgPool for higher-order diff)
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),                # 8x8 (AvgPool for higher-order diff)
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))                           # -> 1x1 (MPS-compatible, works for any input size)
        )
        self.classifier = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(256*1*1, 512),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B,1,32,32)
        feats = self.features(x)
        feats = torch.flatten(feats, 1)
        logits = self.classifier(feats)
        return logits


class SmallScoreNet32x32(nn.Module):
    """
    Very small convnet returning per-pixel score estimate s(x) of shape (B,1,32,32).
    Intended for 32x32 resized MNIST/Fashion-MNIST.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x):
        # returns shape (B,1,32,32) approximating gradient of log-density wrt pixels
        return self.net(x)
