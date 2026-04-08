from typing import Any

from rl4co.envs.common.base import RL4COEnvBase

from CoEKS.models.model import Base

from .policy import CoEKSLightPolicy, CoEKSPolicy
from .CoE import Task


class CoEKS(Base):
    """Original CoEKS model with single variant sampling at each batch"""

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: CoEKSPolicy,
        **kwargs,
    ):
        # print("1111111111111111111111111")
        print(policy)
        super(CoEKS, self).__init__(
            env,
            policy,
            **kwargs,
        )

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):

        out = super(CoEKS, self).shared_step(batch, batch_idx, phase, dataloader_idx)

        # get loss
        loss = out.get("loss", None)

        if loss is not None:
            # Init embeddings
            # Option 1 in the code
            if hasattr(self.policy.encoder.init_embedding, "CoE_loss"):
                CoE_loss_init_embeds = self.policy.encoder.init_embedding.CoE_loss
            else:
                CoE_loss_init_embeds = 0

            # Option 2 in the code
            CoE_loss_layers = 0
            for layer in self.policy.encoder.net.layers:
                if hasattr(layer, "CoE_loss"):
                    CoE_loss_layers += layer.CoE_loss
            # Option 3 in the code
            if hasattr(self.policy.decoder.pointer, "CoE_loss"):
                CoE_loss_decoder = self.policy.decoder.pointer.CoE_loss
            else:
                CoE_loss_decoder = 0

            CoE_loss = CoE_loss_init_embeds + CoE_loss_layers + CoE_loss_decoder
            out["loss"] = loss + CoE_loss

        return out
