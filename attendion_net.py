import torch
import torch.nn as nn

# -------------------------------------------------
# Define Residual Recurrent Blocks for 3D U-Net
# -------------------------------------------------

class RecurrentBlock3D(nn.Module):
    """
    A recurrent convolutional block for 3D data.
    It applies an initial convolution followed by a number of recurrent steps
    where each recurrent step adds its result to the current output.
    """
    def __init__(self, channels, kernel_size=3, recur_num=2):
        super(RecurrentBlock3D, self).__init__()
        padding = kernel_size // 2  # keep spatial dimensions
        self.recur_num = recur_num

        self.initial_conv = nn.Conv3d(channels, channels, kernel_size, padding=padding)
        self.initial_bn   = nn.BatchNorm3d(channels)
        self.relu         = nn.ReLU(inplace=True)

        # Shared convolution for recurrent steps
        self.conv = nn.Conv3d(channels, channels, kernel_size, padding=padding)
        self.bn   = nn.BatchNorm3d(channels)

    def forward(self, x):
        out = self.initial_conv(x)
        out = self.initial_bn(out)
        out = self.relu(out)
        for _ in range(self.recur_num):
            out_res = self.conv(out)
            out_res = self.bn(out_res)
            out_res = self.relu(out_res)
            out = out + out_res  # residual addition
        return out
        
class ResidualRecurrentBlock3D(nn.Module):
    """
    A residual recurrent block that uses a shortcut connection.
    It first creates a shortcut using a 1x1 convolution, then applies two
    recurrent blocks sequentially and adds the shortcut back.
    """
    def __init__(self, in_channels, filters, kernel_size=3, recur_num=2):
        super(ResidualRecurrentBlock3D, self).__init__()
        self.shortcut = nn.Conv3d(in_channels, filters, kernel_size=1, padding=0)
        self.rrb1 = RecurrentBlock3D(filters, kernel_size, recur_num)
        self.rrb2 = RecurrentBlock3D(filters, kernel_size, recur_num)

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.rrb1(shortcut)
        out = self.rrb2(out)
        return out + shortcut
        


class AttentionBlock3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        Parameters:
          F_g: Number of channels in the gating signal (from the decoder).
          F_l: Number of channels in the encoder feature map (skip connection).
          F_int: Number of intermediate channels.
        """
        super(AttentionBlock3D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal (from the decoder, e.g., upsampled feature map)
        # x: encoder feature map from the skip connection
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # Multiply the skip connection by the attention coefficients
        return x * psi

# ------------------------------
# Define the R2-UNet 3D Model with Attention Gates
# ------------------------------


class R2UNet3DWithAttention(nn.Module):
    """
    A Residual Recurrent 3D U-Net (R2-UNet) architecture with Attention Gates
    on the skip connections.
    """
    def __init__(self, in_channels, num_classes, recur_num=2):
        super(R2UNet3DWithAttention, self).__init__()
        print("initialized attendion module")
        # Encoder dropout layers
        self.dropout1 = nn.Dropout3d(0.1)
        self.dropout2 = nn.Dropout3d(0.1)
        self.dropout3 = nn.Dropout3d(0.2)
        self.dropout4 = nn.Dropout3d(0.2)
        self.dropout5 = nn.Dropout3d(0.3)

        # Encoder
        self.c1 = ResidualRecurrentBlock3D(in_channels, 16, recur_num=recur_num)
        self.pool1 = nn.MaxPool3d(2)

        self.c2 = ResidualRecurrentBlock3D(16, 32, recur_num=recur_num)
        self.pool2 = nn.MaxPool3d(2)

        self.c3 = ResidualRecurrentBlock3D(32, 64, recur_num=recur_num)
        self.pool3 = nn.MaxPool3d(2)

        self.c4 = ResidualRecurrentBlock3D(64, 128, recur_num=recur_num)
        self.pool4 = nn.MaxPool3d(2)

        # Bridge
        self.c5 = ResidualRecurrentBlock3D(128, 256, recur_num=recur_num)

        # Decoder
        self.up6 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        # Attention gate for skip connection from c4: channels 128 (both gating and skip) -> intermediate channels = 64
        self.att4 = AttentionBlock3D(F_g=128, F_l=128, F_int=64)
        self.c6 = ResidualRecurrentBlock3D(256, 128, recur_num=recur_num)

        self.up7 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.att3 = AttentionBlock3D(F_g=64, F_l=64, F_int=32)
        self.c7 = ResidualRecurrentBlock3D(128, 64, recur_num=recur_num)

        self.up8 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.att2 = AttentionBlock3D(F_g=32, F_l=32, F_int=16)
        self.c8 = ResidualRecurrentBlock3D(64, 32, recur_num=recur_num)

        self.up9 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.att1 = AttentionBlock3D(F_g=16, F_l=16, F_int=8)
        self.c9 = ResidualRecurrentBlock3D(32, 16, recur_num=recur_num)

        self.out_conv = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.c1(x)
        p1 = self.pool1(c1)
        p1 = self.dropout1(p1)

        c2 = self.c2(p1)
        p2 = self.pool2(c2)
        p2 = self.dropout2(p2)

        c3 = self.c3(p2)
        p3 = self.pool3(c3)
        p3 = self.dropout3(p3)

        c4 = self.c4(p3)
        p4 = self.pool4(c4)
        p4 = self.dropout4(p4)

        c5 = self.c5(p4)
        c5 = self.dropout5(c5)

        # Decoder with Attention Gates
        u6 = self.up6(c5)  # Upsample from bottleneck: shape (B,128,...)
        # Apply attention gate on c4 using u6 as gating signal
        c4_att = self.att4(g=u6, x=c4)
        u6 = torch.cat([u6, c4_att], dim=1)
        c6 = self.c6(u6)
        c6 = self.dropout4(c6)

        u7 = self.up7(c6)
        c3_att = self.att3(g=u7, x=c3)
        u7 = torch.cat([u7, c3_att], dim=1)
        c7 = self.c7(u7)
        c7 = self.dropout3(c7)

        u8 = self.up8(c7)
        c2_att = self.att2(g=u8, x=c2)
        u8 = torch.cat([u8, c2_att], dim=1)
        c8 = self.c8(u8)
        c8 = self.dropout2(c8)

        u9 = self.up9(c8)
        c1_att = self.att1(g=u9, x=c1)
        u9 = torch.cat([u9, c1_att], dim=1)
        c9 = self.c9(u9)
        c9 = self.dropout1(c9)

        outputs = self.out_conv(c9)
        # Note: Do not apply softmax here if using CrossEntropyLoss.
        return outputs
