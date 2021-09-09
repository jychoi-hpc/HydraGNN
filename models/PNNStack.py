import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn import PNAConv, BatchNorm, global_mean_pool

from .Base import Base


class PNNStack(Base):
    def __init__(
        self,
        deg: torch.Tensor,
        input_dim: int,
        output_dim: list,
        output_type: list,
        num_nodes: int,
        hidden_dim: int,
        config_heads: {},
        dropout: float = 0.25,
        num_conv_layers: int = 16,
    ):
        super().__init__()

        aggregators = ["mean", "min", "max", "std"]
        scalers = [
            "identity",
            "amplification",
            "attenuation",
            "linear",
        ]

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_conv_layers = num_conv_layers
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.convs.append(
            PNAConv(
                in_channels=input_dim,
                out_channels=self.hidden_dim,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
        )
        self.batch_norms.append(BatchNorm(self.hidden_dim))
        for _ in range(self.num_conv_layers - 1):
            conv = PNAConv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(self.hidden_dim))

        super()._multihead(output_dim, num_nodes, output_type, config_heads)

    def __str__(self):
        return "PNNStack"
