import torch
import torch.nn as nn
import tqdm
import torch.nn.functional as F

class Soft(nn.Module):
    def __init__(self):
        """
            activate function
        """
        super(Soft, self).__init__()
        self.threshold = nn.Parameter(torch.tensor([0.1]))

    def forward(self, x):
        with torch.no_grad():
            below = x < -self.threshold
            upper = x > self.threshold
        dx = torch.zeros_like(x)
        dx[below] = self.threshold
        dx[upper] = -self.threshold

        return x + dx


class GlobalSimilarity(nn.Module):
    def __init__(self, in_size=None, k=None, pool='mean'):
        """
            similarity between 2 sequences
        :param in_size:   txt_dim == cld_dim
        :param k:         rank of metric
        :param pool:      pooling method
        """
        super(GlobalSimilarity, self).__init__()
        if k is None:
            self.param = None
        else:
            self.param = nn.Parameter(torch.randn(in_size, k) * 1e-1)

        if pool in ['mean', 'avg']:
            self.pool = 'mean'
        else:
            self.pool = 'max'

    def forward(self, txt, cld, cross_sim=False):
        """
        :param cross_sim:   whether to calculate similarity between each pair
        :param txt:     b1, s1, in_size
        :param cld:     b2, s2, in_size
        :return:        b1, b2 if cross_sim else b1
        """
        if cross_sim and txt.dim() == 3:
            return self.forward(txt[:, None, ...], cld[None, ...])
        if self.pool == 'mean':
            txt = torch.mean(txt, dim=-2)
            cld = torch.mean(cld, dim=-2)
        else:
            txt = torch.max(txt, dim=-2)
            cld = torch.max(cld, dim=-2)

        if self.param is None:
            return (txt[..., None, :] @ cld[..., :, None])[..., 0, 0]
        else:
            return (txt[..., None, :] @ (self.param @ self.param.transpose(0, 1)) @ cld[..., :, None])[..., 0, 0]


class GlobalCos(nn.Module):
    def __init__(self, in_size=None, k=None, pool='mean'):
        """
            similarity between 2 sequences
        :param in_size:   txt_dim == cld_dim
        :param k:         rank of metric
        :param pool:      pooling method
        """
        super(GlobalCos, self).__init__()
        if k is None:
            self.param = None
        else:
            self.param = nn.Parameter(torch.randn(in_size, k) * 1e-1)

        if pool in ['mean', 'avg']:
            self.pool = 'mean'
        else:
            self.pool = 'max'

    def forward(self, txt, cld, cross_sim=False):
        """
        :param cross_sim:   whether to calculate similarity between each pair
        :param txt:     b1, s1, in_size
        :param cld:     b2, s2, in_size
        :return:        b1, b2 if cross_sim else b1
        """
        if cross_sim and txt.dim() == 3:
            return self.forward(txt[:, None, ...], cld[None, ...])
        if self.pool == 'mean':
            txt = torch.mean(txt, dim=-2)
            cld = torch.mean(cld, dim=-2)
        else:
            txt = torch.max(txt, dim=-2)[0]
            cld = torch.max(cld, dim=-2)[0]

        # if self.param is not None:
        #     txt = txt @ self.param
        #     cld = cld @ self.param

        # Compute cosine similarity between txt and cld
        cos_sim = F.cosine_similarity(txt.unsqueeze(1), cld.unsqueeze(0), dim=-1)
        # print(f'original cos: {cos_sim}')
        cos_sim = cos_sim.squeeze(1)
        # print(f'squeezed cos:{cos_sim.shape}')
        return cos_sim


class LocalSimilarity(nn.Module):
    def __init__(self, in_size, k, max_txt_len, max_cld_len, n_head, dropout=0.0):
        """
            similarity between 2 tokens
        :param in_size:       txt_dim == cld_dim
        :param k:             qk size
        :param max_txt_len:   max text sequence length
        :param max_cld_len:   max cloud sequence length
        :param n_head:        attention head number
        :param dropout:       dropout
        """
        super(LocalSimilarity, self).__init__()
        self.q = nn.Linear(in_size, k * n_head)
        self.k = nn.Linear(in_size, k * n_head)
        self.time_bias = nn.Parameter(torch.zeros(n_head, max_txt_len, max_cld_len))
        self.n_head = n_head
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, txt, cld, cross_sim=False):
        """
        :param cross_sim:   whether to calculate similarity between each pair
        :param txt:    b1, s1, in_size
        :param cld:    b2, s2, in_size
        :return:       b1, b2, n_head, s1, s2 if cross_sim else b1, n_head, s1, s2
        """
        if cross_sim and txt.dim() == 3:
            return self.forward(txt[:, None, ...], cld[None, ...])

        txt = self.q(txt)  # ..., s1, k
        cld = self.k(cld)  # ..., s2, k

        if txt.dim() == 3:
            b1, s1, _ = txt.shape
            b2, s2, _ = cld.shape
            txt = txt.reshape(b1, s1, self.n_head, -1).transpose(-3, -2)
            cld = cld.reshape(b2, s2, self.n_head, -1).transpose(-3, -2).transpose(-1, -2)
            attn = txt @ cld    # b, n, s1, s2
            attn = attn + self.time_bias[None, :, :s1, :s2]

        else:
            _1, __1, s1, _ = txt.shape
            _2, __2, s2, _ = cld.shape
            txt = txt.reshape(_1, __1, s1, self.n_head, -1).transpose(-3, -2)
            cld = cld.reshape(_2, __2, s2, self.n_head, -1).transpose(-3, -2).transpose(-1, -2)
            attn = txt @ cld  # b1, b2, n, s1, s2

            attn = attn + self.time_bias[None, None, :, :s1, :s2]

        attn = self.dropout(attn)
        return self.gelu(attn)


if __name__ == '__main__':
    model = GlobalSimilarity(10, 2, 'mean')

    x = torch.randn(32, 100, 10)
    y = torch.randn(33, 100, 10)

    print(model(x, y, True).shape)

    model = LocalSimilarity(10, 4, 120, 130, 8, 0.1)

    x = torch.randn(32, 100, 10)
    y = torch.randn(33, 100, 10)

    print(model(x, y, True).shape)

    x = torch.randn(100, 1)
    y = x ** 3

    model = Soft()

    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss_func = nn.L1Loss()

    for epoch in tqdm.trange(100):
        y_hat = model(x)
        loss = loss_func(y_hat, y)

        print(f"epoch: {epoch}, loss: {loss}")

        opt.zero_grad()
        loss.backward()
        opt.step()


