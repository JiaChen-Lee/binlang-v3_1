# Created by Jiachen Li at 2021/1/4 16:45
import torch
import torch.nn as nn
from einops import rearrange, repeat


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, d_model, nhead, num_encoder_layers, num_decoder_layers, channels=3, dim_head=64,
                 dropout=0., emb_dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        self.patch_to_embedding = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.Transformer(d_model=512,
                                          nhead=8,
                                          num_encoder_layers=6,
                                          num_decoder_layers=6,
                                          dim_feedforward=2048,
                                          dropout=0.1)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, img, mask=None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)
