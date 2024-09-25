import torch.nn as nn
import torch
import warnings


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, *args):
        """
        :param in_ch:    in channel
        :param out_ch:   out channel
        :param kernel_size:   kernel size
        :param stride:      stride
        :param args:       useless
        """
        super(ResBlock, self).__init__()
        self.args = args
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=1)
        )
        self.shortcut = nn.Identity() if in_ch == out_ch and stride == 1 else nn.Conv2d(in_ch, out_ch, 3, stride, 1)
        self.norm = nn.GroupNorm(out_ch // 2, out_ch)

    def forward(self, x):
        """
        :param x:   size(batch_size, n_head, s1, s2)
        :return:
        """
        return self.norm(self.conv(x) + self.shortcut(x))


class DownSample(nn.Module):
    def __init__(self, stride, kernel_size=None):
        """
        :param stride:
        :param kernel_size:
        """
        super(DownSample, self).__init__()
        if kernel_size is None:
            kernel_size = 3 if stride < 3 else stride + 3
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.pool(x)


class GlobalSimilarity(nn.Module):
    def __init__(self, n_head):
        super(GlobalSimilarity, self).__init__()
        self.n_head = n_head

    def forward(self, x, has_act=False):
        """
        :param has_act:
        :param x:   size(batch_size, n_head, s1, s2)
        :return:    size(batch_size, 1)
        """
        assert x.dim() == 4
        warnings.warn("forward method has not been written")


class LinearGlobalSimilarity(GlobalSimilarity):
    def __init__(self, n_head, dropout=0.1):
        super(LinearGlobalSimilarity, self).__init__(n_head)
        self.dropout = nn.Dropout(dropout)
        self.tail = nn.Sequential(
            nn.Linear(n_head, 1),
            nn.Sigmoid()
        )
        self.act = nn.GELU()

    def forward(self, x, has_act=False):
        x = self.dropout(x)
        if not has_act:
            x = self.act(x)
        out = torch.mean(x, dim=[2, 3])
        return self.tail(out)


class ConvGlobalSimilarity(GlobalSimilarity):
    def __init__(self, n_head, dropout=0.1, base_block=nn.Conv2d, channels=None):
        """
        :param n_head:
        :param dropout:
        :param base_block:   nn.Conv2d or ResBlock
        :param channels:      [layer1.channel, layer2.channel]
        """
        super(ConvGlobalSimilarity, self).__init__(n_head)
        self.dropout = nn.Dropout(dropout)
        if channels is None:
            channels = [4, 8]

        lst = []
        now_ch = n_head
        for ch in channels:
            lst.append(base_block(now_ch, ch, 3, 1, 1))
            lst.append(DownSample(4, kernel_size=8))  # kernel size too large
            now_ch = ch
        lst.append(nn.AdaptiveAvgPool2d(1))
        self.conv = nn.Sequential(*lst)

        self.tail = nn.Sequential(
            nn.Linear(now_ch, 1),
            nn.Sigmoid()
        )

        self.act = nn.GELU()

    def forward(self, x, has_act=False):
        if not has_act:
            x = self.act(x)

        out = self.dropout(x)
        out = self.conv(out)[:, :, 0, 0]
        out = self.tail(out)
        return out


if __name__ == '__main__':
    x = torch.randn(32, 4, 128, 128)
    model = LinearGlobalSimilarity(4)
    print(model(x).shape)
    model = ConvGlobalSimilarity(4, base_block=ResBlock)
    print(model(x).shape)
