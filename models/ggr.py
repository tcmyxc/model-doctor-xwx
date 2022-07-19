import torch
from torch import nn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0.):
        super().__init__()
        self.scale = qk_scale or dim ** -0.5
        self.norm1q = nn.LayerNorm(dim)
        self.norm1k = nn.LayerNorm(dim)

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, qx, kx):
        qx = qx.unsqueeze(1)
        kx = kx if len(kx.shape) == 3 else kx.unsqueeze(1)
        # qx:[Bq, 1, C]
        # kx:[Bk, Nk, C]
        assert qx.shape[-1] == kx.shape[-1] and qx.shape[1] == 1

        q = self.wq(self.norm1q(qx))
        k = self.wk(self.norm1k(kx))
        v = kx
        attn = torch.einsum('qoc,knc->qkn', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        x = torch.einsum('knc,qkn->qkc', v, attn)

        idx = cos(qx, x).argmax(-1)
        return x[:, idx, :][0]


class GGR(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.mlp = Mlp(dim)
        self.attn = Attention(dim)

    def forward(self, x, kx):
        out1 = self.mlp(x)
        out2 = self.attn(x, kx)

        return out1 + out2


if __name__ == "__main__":
    ggr = GGR()
    x = torch.rand((8, 64))
    qx = torch.rand((10, 64))
    print(ggr(x, qx))

