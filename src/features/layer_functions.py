import torch
import torch.nn as nn
import torch.nn.functional as F


class CBR(nn.Module):
    def __init__(self, in_ch, out_ch, circular_padding, filter_size=3, padding_size=1, bias=True, mode="CBR"):
        super(CBR, self).__init__()

        self.circular_padding = circular_padding
        self.filter_size = filter_size
        self.padding_size = padding_size
        self.bias = bias
        self.mode = mode
        if self.circular_padding:
            self.padding_version = (self.padding_size, 0)
        else:
            self.padding_version = self.padding_size

        if self.mode == "CBR":
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=self.filter_size, padding=self.padding_version, bias=self.bias),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
            )
        elif self.mode == "CRB":
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=self.filter_size, padding=self.padding_version, bias=self.bias),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        if self.circular_padding:
            x = F.pad(x, (1, 1, 0, 0), mode="circular")
            x = self.conv(x)
        else:
            x = self.conv(x)
        return x


class up_CBR(nn.Module):
    def __init__(self, in_ch, out_ch, circular_padding, filter_size=3, padding_size=1, bias=True, mode="CBR"):
        super(up_CBR, self).__init__()

        self.circular_padding = circular_padding
        self.filter_size = filter_size
        self.padding_size = padding_size
        self.bias = bias
        self.mode = mode

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2, groups=in_ch // 2)
        self.conv = CBR(in_ch, out_ch, self.circular_padding, self.filter_size, self.padding_size, self.bias, self.mode)

    def forward(self, x1, x2):
        # TODO: cite this
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class down_CBR(nn.Module):
    def __init__(self, in_ch, out_ch, circular_padding, filter_size=3, padding_size=1, bias=True):
        super(down_CBR, self).__init__()

        self.circular_padding = circular_padding
        self.filter_size = filter_size
        self.padding_size = padding_size
        self.bias = bias

        self.pooling = nn.MaxPool2d(2)
        self.conv = CBR(in_ch, out_ch, self.circular_padding, self.filter_size, self.padding_size, self.bias)

    def forward(self, x):
        x = self.pooling(x)
        x = self.conv(x)
        return x


class BlockDrop(nn.Module):
    def __init__(self, drop_p=0.5, block_size=7):
        super(BlockDrop, self).__init__()
        self.drop_prob = drop_p
        self.block_size = block_size

    def forward(self, x):
        gamma = self.drop_prob / (self.block_size**2)
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float().to(x.device)

        block_mask = compute_block_mask(mask, 7)

        out = x * block_mask[:, None, :, :]
        out = out * block_mask.numel() / block_mask.sum()
        return out


def compute_block_mask(mask, block_size):
    block_mask = F.max_pool2d(
        input=mask[:, None, :, :], kernel_size=(block_size, block_size), stride=(1, 1), padding=block_size // 2
    )

    if block_size % 2 == 0:
        block_mask = block_mask[:, :, :-1, :-1]

    block_mask = 1 - block_mask.squeeze(1)

    return block_mask


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, circular_padding=False):
        super().__init__()

        self.circular_padding = circular_padding
        self.stride = stride

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.conv1 = CBR(out_channels, out_channels, self.circular_padding, filter_size=3, padding_size=1, bias=False)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        if self.stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=self.stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        x = nn.ReLU(self.conv2(self.conv1(self.conv0(x))) + self.shortcut(x))
        return x
