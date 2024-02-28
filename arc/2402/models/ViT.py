import torch
from torch import nn
from torch.nn.modules import loss
from torchvision.models import ViT_B_16_Weights, vit_b_16
import lightning as L

# local import
from niche import Niche_Lightning


class Niche_ViT(Niche_Lightning):
    def __init__(
        self,
        n_out=1,
        n_blocks=4,
        weights="IMAGENET1K_SWAG_E2E_V1",
        # base
        lr=1e-3,
        optimizer="Adam",
        loss="MSE",
    ):
        # init
        super().__init__(loss=loss, optimizer=optimizer, lr=lr)

        # backbone ViT_b_16
        if weights == "IMAGENET1K_SWAG_E2E_V1":
            weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        self.model = vit_b_16(weights=weights)

        # MLP: check 0504_ViT_arch.ipynb
        in_features = self.model.heads.head.in_features
        hidden_features = 512
        nlp_blocks = []
        for _ in range(n_blocks):
            nlp_blocks.extend(
                [
                    nn.Linear(in_features, hidden_features),
                    nn.GELU(),
                ]
            )
            in_features = hidden_features
            hidden_features = hidden_features // 2
        # add final layer
        nlp_blocks.extend(
            [
                nn.Linear(in_features, n_out),
            ]
        )
        # replace head
        self.model.heads = nn.Sequential(*nlp_blocks)

    def forward(self, x):
        return self.model(x)
