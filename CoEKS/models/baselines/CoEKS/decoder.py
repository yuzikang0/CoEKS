import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl4co.envs import RL4COEnvBase
from rl4co.models.nn.attention import PointerAttention

from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
from CoEKS.models.baselines.CoEKS.rl4co_decoder import AttentionModelDecoder

from rl4co.utils.pylogger import get_pylogger
from torch.nn.functional import scaled_dot_product_attention

from CoEKS.models.env_embeddings.mtvrp.context import MTVRPContextEmbedding

from .CoE import Task

log = get_pylogger(__name__)

class MLP(nn.Module):
    def __init__(self, input_size=128, output_size=128, hidden_size=512):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PointerAttentionCoE(PointerAttention):
    """
    CoEKS replaces the project_out to obtain the glimpse
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feedforward_hidden: int = 512,
        mask_inner: bool = True,
        out_bias: bool = False,
        check_nan: bool = True,
        sdpa_fn=None,
        num_experts=5,
        routing_method="input_choice",
        routing_level="node",
        topk=2,
        hierarchical_gating=False,
        temperature=1.0,
        sign_task=False
    ):
        nn.Module.__init__(self)
        self.num_heads = num_heads
        self.mask_inner = mask_inner
        self.sdpa_fn = sdpa_fn if sdpa_fn is not None else scaled_dot_product_attention
        self.check_nan = check_nan
        self.routing_method = routing_method
        self.routing_level = routing_level
        self.topk = topk
        self.hierarchical_gating = hierarchical_gating
        self.temperature = temperature
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=out_bias)
        self.IDT = nn.Linear(4, embed_dim, bias=out_bias)
        self.mlp = MLP(input_size=embed_dim, output_size=embed_dim, hidden_size=512)
       


    def forward(self, query, context_embedding, key, value, logit_key, active_experts, attn_mask=None):
        # Compute inner multi-head attention with no projections.
        heads = self._inner_mha(query, key, value, attn_mask)
        mh_atten_out = self._project_out(heads, attn_mask)
    
        # CoE loss
        coe_loss = 0


        '''RELD'''
        split_sizes = [query.size()[-1], context_embedding.size(-1) - query.size()[-1]]
        cur_node_embedding, state_embedding = torch.split(context_embedding, split_sizes, dim=-1)
        idt = self.IDT(state_embedding)
        mh_atten_out_IDT = mh_atten_out + cur_node_embedding + idt
        mlpout = self.mlp(mh_atten_out_IDT)
        glimpse = mh_atten_out_IDT + mlpout
        '''RELD'''

        logits = (torch.bmm(glimpse, logit_key.squeeze(-2).transpose(-2, -1))).squeeze(
            -2
        ) / math.sqrt(glimpse.size(-1))

        if self.check_nan:
            assert not torch.isnan(logits).any(), "Logits contain NaNs"

        self.coe_loss = coe_loss

        return logits


class CoEKSDecoder(AttentionModelDecoder):
    """
    TODO
    Note that the real change is the pointer attention
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        env_name: str = "tsp",
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        linear_bias: bool = False,
        use_graph_context: bool = True,
        check_nan: bool = True,
        sdpa_fn: callable = None,
        num_experts=5,
        routing_method="input_choice",
        routing_level="node",
        topk=2,
        CoE_loc=["enc0", "enc1", "enc2", "enc3", "enc4", "enc5", "dec"],
        hierarchical_gating=False,
    ):
        super().__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0

        if context_embedding is None:
            log.info("Using default MTVRPContextEmbedding")
            context_embedding = MTVRPContextEmbedding(embed_dim)
        self.context_embedding = context_embedding

        if dynamic_embedding is None:
            log.info("Using default StaticEmbedding")
            self.dynamic_embedding = StaticEmbedding()
        self.is_dynamic_embedding = (
            False if isinstance(self.dynamic_embedding, StaticEmbedding) else True
        )

        self.project_node_embeddings = nn.Linear(
            embed_dim, 3 * embed_dim, bias=linear_bias
        )
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.use_graph_context = use_graph_context


        self.pointer = PointerAttentionCoE(
            embed_dim,
            num_heads,
            feedforward_hidden=512,
            mask_inner=mask_inner,
            out_bias=out_bias_pointer_attn,
            check_nan=check_nan,
            sdpa_fn=sdpa_fn,
            num_experts=5,
            routing_method=routing_method,
            routing_level=routing_level,
            topk=topk,
            hierarchical_gating=hierarchical_gating,
            sign_task=False if f"dec" not in CoE_loc else True
        )

