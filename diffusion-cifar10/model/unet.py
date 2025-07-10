import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, dropout=0.):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out)
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, dim_out),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim_out, dim_out, 3, padding=1)
        )
        
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h += self.time_mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)  # (b, h*w, c)
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, h * w, self.num_heads, c // self.num_heads).transpose(1, 2), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, h * w, c)
        out = self.to_out(out)
        
        out = out.permute(0, 2, 1).view(b, c, h, w)
        return out

class UNet(nn.Module):
    def __init__(self, dim=64, dim_mults=(1, 2, 4, 8), channels=3, dropout=0.1):
        super().__init__()
        
        # Time embedding
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(channels, dim, 7, padding=3)
        
        # Downsampling layers
        self.downs = nn.ModuleList([])
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            
            self.downs.append(nn.ModuleList([
                ResBlock(dim_in, dim_out, time_dim, dropout=dropout),
                ResBlock(dim_out, dim_out, time_dim, dropout=dropout),
                AttentionBlock(dim_out) if dim_out <= 256 else nn.Identity(),
                nn.Conv2d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity()
            ]))
        
        # Middle layers
        mid_dim = dims[-1]
        self.mid_block1 = ResBlock(mid_dim, mid_dim, time_dim, dropout=dropout)
        self.mid_attn = AttentionBlock(mid_dim)
        self.mid_block2 = ResBlock(mid_dim, mid_dim, time_dim, dropout=dropout)
        
        # Upsampling layers
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            
            self.ups.append(nn.ModuleList([
                ResBlock(dim_out * 2, dim_in, time_dim, dropout=dropout),
                ResBlock(dim_in, dim_in, time_dim, dropout=dropout),
                AttentionBlock(dim_in) if dim_in <= 256 else nn.Identity(),
                nn.ConvTranspose2d(dim_in, dim_in, 4, 2, 1) if not is_last else nn.Identity()
            ]))
        
        # Final layers
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, channels, 3, padding=1)
        )

    def forward(self, x, time):
        t = self.time_mlp(time)
        
        x = self.init_conv(x)
        
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        
        # Middle processing with attention
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
        
        return self.final_conv(x) 