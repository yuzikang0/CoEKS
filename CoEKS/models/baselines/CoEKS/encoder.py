import torch
import torch.nn as nn

from rl4co.envs import RL4COEnvBase
from rl4co.models.nn.attention import MultiHeadAttention
from rl4co.models.nn.graph.attnnet import GraphAttentionNetwork
from rl4co.models.nn.ops import Normalization
# from rl4co.models.zoo.am.encoder import AttentionModelEncoder
from CoEKS.models.baselines.CoEKS.rl4co_encoder import AttentionModelEncoder
from rl4co.utils.pylogger import get_pylogger
from torch import Tensor

from CoEKS.models.env_embeddings.mtvrp.init import MTVRPInitEmbedding

from .CoE import Task

import csv
from pathlib import Path
import os

log = get_pylogger(__name__)


class CoEKSInitEmbedding(MTVRPInitEmbedding):
    def __init__(
        self,
        embed_dim=128,
        num_experts=5,
        routing_method="input_choice",
        routing_level="node",
        topk=2,
        bias=False,  # False
        **kw,
    ):  # node: linear bias should be false in order not to influence the embedding if
        super(CoEKSInitEmbedding, self).__init__(embed_dim, bias, **kw)

        # If CoE is provided, we re-initialize the projections with CoE
        if num_experts > 0:
            print("CoE in init embedding initializing")
            self.project_global_feats = Task(
                input_size=2,
                output_size=embed_dim,
                num_experts=num_experts,
                k=topk,
                T=1.0,
                noisy_gating=True,
                routing_level=routing_level,
                routing_method=routing_method,
                CoE_model="Linear",
            )
            self.project_customers_feats = Task(
                input_size=7,
                output_size=embed_dim,
                num_experts=num_experts,
                k=topk,
                T=1.0,
                noisy_gating=True,
                routing_level=routing_level,
                routing_method=routing_method,
                CoE_model="Linear",
            )

    def forward(self, td):
        # Global (batch, 1, 2) -> (batch, 1, embed_dim)
        global_feats = td["locs"][:, :1, :]

        # Customers (batch, N, 5) -> (batch, N, embed_dim)
        # note that these feats include the depot (but unused) so we exclude the first node
        cust_feats = torch.cat(
            (
                td["demand_linehaul"][..., 1:, None],
                td["demand_backhaul"][..., 1:, None],
                td["time_windows"][..., 1:, :],
                td["service_time"][..., 1:, None],
                td["locs"][:, 1:, :],
            ),
            -1,
        )

        # If some features are infinity (e.g. distance limit is inf because of no limit), replace with 0 so that it does not affect the embedding
        global_feats = torch.nan_to_num(global_feats, nan=0.0, posinf=0.0, neginf=0.0)
        cust_feats = torch.nan_to_num(cust_feats, nan=0.0, posinf=0.0, neginf=0.0)

        # CoE loss is 0 if layer is not CoE
        CoE_loss_global, CoE_loss_cust = 0, 0
        if isinstance(self.project_global_feats, Task):
            global_embeds, CoE_loss_global = self.project_global_feats(global_feats)
        else:
            global_embeds = self.project_global_feats(global_feats)
        if isinstance(self.project_customers_feats, Task):
            cust_embeds, CoE_loss_cust = self.project_customers_feats(cust_feats)
        else:
            cust_embeds = self.project_customers_feats(cust_feats)
        self.CoE_loss = CoE_loss_global + CoE_loss_cust
        return torch.cat((global_embeds, cust_embeds), -2)


class MultiHeadAttentionLayerCoE(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        feedforward_hidden: int = 512,
        normalization="instance",
        sdpa_fn=None,
        num_experts=5,
        routing_method="input_choice",
        routing_level="node",
        topk=2,
        sign_task=False
    ):
        super(MultiHeadAttentionLayerCoE, self).__init__()

        if num_experts > 0:
            print("CoE in MultiHeadAttentionLayer initializing")
            dense_net = Task(
                input_size=embed_dim,
                output_size=embed_dim,
                num_experts=num_experts,
                hidden_size=feedforward_hidden,
                k=topk,
                T=1.0,
                noisy_gating=True,
                routing_level=routing_level,
                routing_method=routing_method,
                CoE_model="MLP",
                sign=sign_task
            )
        else:
            dense_net = nn.Sequential(
                nn.Linear(embed_dim, feedforward_hidden),
                nn.ReLU(),
                nn.Linear(feedforward_hidden, embed_dim),
            )

        self.mha = MultiHeadAttention(embed_dim, num_heads, sdpa_fn=sdpa_fn)
        self.norm1 = Normalization(embed_dim, normalization)
        self.dense = dense_net
        self.norm2 = Normalization(embed_dim, normalization) 

    def forward(self, x: Tensor, activate_index) -> Tensor:
        out_mha = self.mha(x)
        h = out_mha + x  # skip connection
        h = self.norm1(h)
        CoE_loss = 0
        if isinstance(self.dense, Task):
            out_dense, CoE_loss = self.dense(h, activate_index)
        else:
            out_dense = self.dense(h)


        self.CoE_loss = CoE_loss

        h = out_dense + h 
        h = self.norm2(h)
        return h


class GraphAttentionNetworkCoEKS(GraphAttentionNetwork):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        num_layers: int,
        normalization: str = "instance",
        feedforward_hidden: int = 512,
        sdpa_fn=None,
        CoE_loc=["enc0", "enc1", "enc2", "enc3", "enc4", "enc5", "dec"],
        num_experts=5,
        routing_method="input_choice",
        routing_level="node",
        topk=2,
    ):
        nn.Module.__init__(self)

        self.layers = nn.ModuleList(
            [
                MultiHeadAttentionLayerCoE(
                    embed_dim,
                    num_heads,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                    sdpa_fn=sdpa_fn,
                    num_experts=num_experts,
                    routing_method=routing_method,
                    routing_level=routing_level,
                    topk=topk,
                    sign_task=False if f"enc{i}" not in CoE_loc else True
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, activate_index, mask=None) -> Tensor:
        """Forward pass of the encoder

        Args:
            x: [batch_size, graph_size, embed_dim] initial embeddings to process
            mask: [batch_size, graph_size, graph_size] mask for the input embeddings. Unused for now.
        """
        assert mask is None, "Mask not yet supported!"


        h = x

        for i, layer in enumerate(self.layers):
            h = layer(h, activate_index)

        return h


class CoEKSEncoder(AttentionModelEncoder):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        normalization: str = "instance",
        feedforward_hidden: int = 512,
        env_name="mtvrp",
        sdpa_fn=None,
        init_embedding=None,
        num_experts=5,
        routing_method="input_choice",
        routing_level="node",
        topk=2,
        CoE_loc=["enc0", "enc1", "enc2", "enc3", "enc4", "enc5", "dec"],
        **unused,
    ):

        nn.Module.__init__(self)

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        assert self.env_name == "mtvrp", "Only mtvrp is supported for CoEKS"

        # assert init_embedding is None, "init embedding is manually set in CoEKS"

        # Initialize raw features only if provided
        if "raw" in CoE_loc:
            num_experts_init = num_experts
        else:
            num_experts_init = 0

        if not init_embedding:
            init_embedding = CoEKSInitEmbedding(
                embed_dim,
                num_experts=num_experts_init,
                routing_method=routing_method,
                routing_level=routing_level,
                topk=topk,
            )
        else:
            if num_experts_init > 0:
                log.warning("CoE requested for init embedding but already provided")
        self.init_embedding = init_embedding

        self.net = GraphAttentionNetworkCoEKS(
            num_heads,
            embed_dim,
            num_layers,
            normalization,
            feedforward_hidden,
            sdpa_fn=sdpa_fn,
            CoE_loc=CoE_loc,
            num_experts=num_experts,
            routing_method=routing_method,
            routing_level=routing_level,
            topk=topk,
        )
