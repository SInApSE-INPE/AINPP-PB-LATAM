import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name):
    name = name.lower()
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    elif name == 'elu':
        return nn.ELU(inplace=True)
    elif name == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation: {name}")

def get_norm(name, channels):
    name = name.lower()
    if name == 'batch':
        return nn.BatchNorm2d(channels)
    elif name == 'instance':
        return nn.InstanceNorm2d(channels)
    elif name == 'group':
        return nn.GroupNorm(num_groups=8, num_channels=channels) # Default 8 groups, modify if needed
    elif name == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Unknown normalization: {name}")

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, padding=1, activation='relu', norm='batch', dropout=0.0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        layers = []
        # Conv 1
        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(get_norm(norm, mid_channels))
        layers.append(get_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
            
        # Conv 2
        layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(get_norm(norm, out_channels))
        layers.append(get_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, pooling='max', **kwargs):
        super().__init__()
        if pooling == 'max':
            pool = nn.MaxPool2d(2)
        elif pooling == 'avg':
            pool = nn.AvgPool2d(2)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
            
        self.maxpool_conv = nn.Sequential(
            pool,
            DoubleConv(in_channels, out_channels, **kwargs)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, **kwargs):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, **kwargs)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, **kwargs)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()
        
        # Parse config
        if hasattr(config, 'model'):
             conf = config.model
        else:
            conf = config

        self.n_channels = conf.in_channels
        self.n_classes = conf.out_channels
        self.bilinear = conf.get('bilinear', True)
        self.features = conf.get('features', [64, 128, 256, 512])
        
        # Conv params
        dropout = conf.get('dropout', 0.0)
        activation = conf.get('activation', 'relu')
        norm = conf.get('normalization', 'batch')
        kernel_size = conf.get('kernel_size', 3)
        padding = conf.get('padding', 1)
        pooling = conf.get('pooling', 'max')
        
        conv_kwargs = {
            'activation': activation,
            'norm': norm,
            'dropout': dropout,
            'kernel_size': kernel_size,
            'padding': padding
        }

        self.inc = DoubleConv(self.n_channels, self.features[0], **conv_kwargs)
        
        # Dynamic Down layers
        self.downs = nn.ModuleList()
        in_ch = self.features[0]
        for out_ch in self.features[1:]:
            self.downs.append(Down(in_ch, out_ch, pooling=pooling, **conv_kwargs))
            in_ch = out_ch
            
        factor = 2 if self.bilinear else 1
        
        # Dynamic Up layers (reverse order)
        self.ups = nn.ModuleList()
        # Features: [64, 128, 256, 512]
        # Down path goes: 64->128->256->512
        # Usually typical UNet has bottleneck then Up. 
        # But this implementation treated last down as bottleneck?
        # Let's align with original:
        # inc: -> 64
        # down1: 64 -> 128
        # down2: 128 -> 256
        # down3: 256 -> 512
        # down4 (bottleneck): 512 -> 1024 // factor
        # Original code had explicit down1..down4. 
        # To make it dynamic, we need to decide if 'features' includes the bottleneck.
        # Let's assume 'features' are the encoder stages.
        # Original: [64, 128, 256, 512] -> Bottle: 1024.
        
        # Wait, the original code had:
        # down1: 64->128
        # ...
        # down3: 256->512
        # down4: 512->1024 (bottleneck)
        
        # So if features=[64, 128, 256, 512], we interpret this as:
        # Encoder output channels.
        # The last element is NOT the bottleneck output, but the input to the bottleneck?
        # Original:
        # inc: n_channels -> 64
        # down1: 64 -> 128
        # down2: 128 -> 256
        # down3: 256 -> 512
        # down4: 512 -> 1024 (Bottleneck)
        
        # So default features list length 4 implies 4 downsampling steps.
        # The last downsampling produces the bottleneck features.
        
        # To replicate orig behavior:
        # We need a bottleneck layer that doubles channels one last time.
        
        self.downs = nn.ModuleList()
        in_c = self.features[0]
        for f in self.features[1:]:
            self.downs.append(Down(in_c, f, pooling=pooling, **conv_kwargs))
            in_c = f
            
        # Bottleneck
        # Original: down4(512, 1024 // factor)
        self.bottleneck = Down(self.features[-1], (self.features[-1]*2) // factor, pooling=pooling, **conv_kwargs)
        
        # Ups
        # Original up1: (1024, 512 // factor)
        # It takes bottleneck output and skip connection from last encoder layer.
        # Bottleneck out: (features[-1]*2) // factor
        # Skip: features[-1]
        # Out: features[-1] // factor
        
        self.ups = nn.ModuleList()
        
        # Reversed features for decoding
        # Skip connections come from valid features: [64, 128, 256, 512]
        # We iterate backwards.
        
        # Start with bottleneck output channels
        current_ch = (self.features[-1]*2) // factor
        
        for i, f in enumerate(reversed(self.features)):
            # Up(in_channels, out_channels)
            # in_channels must match the concatenation of upsampled input + skip connection.
            # Upsample doesn't change channels if mode is bilinear (usually). 
            # But the Up module handles the logic.
            # If bilinear=True, Up module expects in_channels to be the SUM of skip + upsampled.
            
            # Input from below: current_ch
            # Input from skip: f
            
            # However, the Up class implementation of `DoubleConv` uses `in_channels` to define the first conv input.
            # The `Up.forward` concatenates `x2` (skip) and `x1` (upsampled).
            # So channel count is `skip.channels + upsampled.channels`.
            # Upsampled channels = current_ch.
            # Skip channels = f.
            
            in_ch_total = current_ch + f
            out_ch = f // factor
            
            # Special handling for the last block to match original dimensions if needed?
            # Original: up4 goes to 64. features[0] is 64. 64//2 = 32?
            # Original code: 
            # up4 = Up(128, 64) -> in=128 (64+64), out=64.
            # features[0]=64. factor=1 (original default bilinear=True but factor logic applied differently?)
            # In orig: factor=2 if bilinear.
            # up4: Up(128, 64, bilinear). conv input 128. out 64.
            # My logic: f=64. current_ch=64 (from prev UP). in_ch_total = 64+64=128. 
            # out_ch = 64 // 2 = 32.
            # THIS IS WRONG. The output of the Up block should effectively restore the channel count of the skip layer (or close to it)
            # to prepare for the next one, OR match features[0] at the end.
            
            # Original up4 output is 64. features[0] is 64.
            # So out_ch should be `f`?
            # If `out_ch = f`:
            # Up(128, 64). DoubleConv(128, 64, mid=64). 
            # This matches original.
            
            # But wait, factor only affecting bottleneck? 
            # Original:
            # down4(512, 1024//2 = 512).  <-- Bottleneck out is 512.
            # up1(1024, 256). Input to up1 is cat(512, 512) = 1024. Out 256.
            # My logical bottleneck out: 512.
            # My f (skip): 512.
            # in_total: 1024.
            # out_ch: f // 2 = 256.
            # Next iter: current=256. f=256. in=512. out=128.
            # ...
            # Last iter: current=?. f=64. in=?. out=32?
            
            # Original OutConv takes 64.
            
            # Let's trust 'f // factor' for inner layers, but maybe we shouldn't divide by factor for the output channels?
            # Actually, `Up` class divides `in_channels // 2` for `mid_channels`.
            
            # If I use `out_ch = f`, then:
            # Iter 1: in=1024. out=512.
            # Iter 2: in=512+256=768? No.
            # If Out=512. Next current=512. Skip=256. Sum=768.
            # This diverges from powers of 2.
            
            # The issue is `factor`.
            # If bilinear, we lose channels via interpolation or just don't increase them?
            # If bilinear, we want to halve channels at each step to eventually reach features[0].
            
            # Let's look at standard UNet sizes:
            # 64->128->256->512->1024 (Bot)
            # Up: 1024->512 (cat 512+512).
            # Up: 512->256 (cat 256+256).
            # Up: 256->128 (cat 128+128).
            # Up: 128->64 (cat 64+64).
            
            # So `out_ch` should be `f`. WITHOUT factor division?
            # If bilinear, factor=2.
            # Bottleneck out = 512. (1024//2).
            # Up1 input: 512 (bot) + 512 (skip) = 1024.
            # Up1 output target: 256. (So 512 // 2).
            
            # So `out_ch` should be `f // factor`?
            # 512 // 2 = 256. Correct.
            # Next: current=256. Skip=256.
            # Up2 input: 256+256 = 512.
            # Up2 output: 256 // 2 = 128. Correct.
            
            # Last one: f=64.
            # current=64 (from previous 128//2).
            # Skip=64.
            # Input: 128.
            # Output: 64 // 2 = 32?
            # But we want 64 for OutConv.
            
            # The original code sets `up4 = Up(128, 64)`.
            # Note 'factor' was NOT used in the output channel calc for up4 in original code!
            # `self.up4 = Up(128, 64, bilinear)`
            
            # So for the very last layer (corresponding to features[0]), we should probably NOT divide by factor, or explicitly target features[0].
            
            if i == len(self.features) - 1:
                out_ch = f
            else:
                out_ch = f // factor
            
            self.ups.append(Up(in_ch_total, out_ch, self.bilinear, **conv_kwargs))
            current_ch = out_ch
            
        self.outc = OutConv(current_ch, self.n_classes)

    def forward(self, x):
        x = self.inc(x)
        skips = [x]
        
        for down in self.downs:
            x = down(x)
            skips.append(x)
            
        x = self.bottleneck(x)
        
        # skips has [inc_out, down1_out, ..., downN_out]
        # bottleneck uses downN_out.
        # Ups need to consume skips in reverse.
        
        for i, up in enumerate(self.ups):
            skip = skips[-(i+1)]
            x = up(x, skip)
            
        logits = self.outc(x)
        return logits
