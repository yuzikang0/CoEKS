import torch
from typing import Tuple, Union

import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive import AutoregressiveEncoder
from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.nn.graph.attnnet import GraphAttentionNetwork


class AttentionModelEncoder(AutoregressiveEncoder):

    def __init__(
        self,
        embed_dim: int = 128,
        init_embedding: nn.Module = None,
        env_name: str = "tsp",
        num_heads: int = 8,
        num_layers: int = 3,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        net: nn.Module = None,
        sdpa_fn = None,
        coe_kwargs: dict = None,
    ):
        super(AttentionModelEncoder, self).__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name

        self.init_embedding = (
            env_init_embedding(self.env_name, {"embed_dim": embed_dim})
            if init_embedding is None
            else init_embedding
        )

        self.net = (
            GraphAttentionNetwork(
                num_heads,
                embed_dim,
                num_layers,
                normalization,
                feedforward_hidden,
                sdpa_fn=sdpa_fn,
                coe_kwargs=coe_kwargs,
            )
            if net is None
            else net
        )

    def forward(
        self, td: TensorDict, mask: Union[Tensor, None] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass of the encoder.
        Transform the input TensorDict into a latent representation.

        Args:
            td: Input TensorDict containing the environment state
            mask: Mask to apply to the attention

        Returns:
            h: Latent representation of the input
            init_h: Initial embedding of the input
        """

        init_h = self.init_embedding(td)
        
        dim0 = torch.ones(init_h.size()[0], 1, device=td.device)  # C

        dim1 = td["open_route"].int()  # O
        dim2 = (td["demand_backhaul"] != 0).any(dim=1, keepdim=True).int()  # B
        dim3 = (~torch.isinf(td["distance_limit"])).int()  # L
        dim4 = (td["service_time"] != 0).any(dim=1, keepdim=True).int()  # TW
        
        self.active_experts = torch.cat([dim0, dim1, dim2, dim3, dim4], dim=1)

        # Process embedding
        h = self.net(init_h, self.active_experts, mask)  # Task

        return h, init_h