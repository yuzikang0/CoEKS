import torch
import torch.nn as nn


class MTVRPInitEmbedding(nn.Module):
    """Initial embedding MTVRP.


    Customer features:
        - locs: x, y euclidean coordinates
        - demand_linehaul: demand of the nodes (delivery) (C)
        - demand_backhaul: demand of the nodes (pickup) (B)
        - time_windows: time window (TW)
        - durations: duration of the nodes (TW)

    Global features:
        - loc: x, y euclidean coordinates of depot
    """

    def __init__(
        self, embed_dim=128, bias=False, global_feat_d=2, **kw
    ):  # node: linear bias should be false in order not to influence the embedding if
        super(MTVRPInitEmbedding, self).__init__()

        # Depot feats (includes global features): x, y, distance, backhaul_class, open_route
        global_feat_dim = global_feat_d
        self.project_global_feats = nn.Linear(global_feat_dim, embed_dim, bias=bias)

        # Customer feats: x, y, demand_linehaul, demand_backhaul, time_window_early, time_window_late, durations
        customer_feat_dim = 7
        self.project_customers_feats = nn.Linear(customer_feat_dim, embed_dim, bias=bias)

        self.embed_dim = embed_dim

    def _global_feats(self, td):
        raise NotImplementedError("_global_feats should be overridden by subclasses")

    def forward(self, td):
        global_feats = self._global_feats(td)

        # Global (batch, 1, 2) -> (batch, 1, embed_dim)
        # global_feats = td["locs"][:, :1, :]

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
        global_embeddings = self.project_global_feats(
            global_feats
        )  # [batch, 1, embed_dim]
        cust_embeddings = self.project_customers_feats(
            cust_feats
        )  # [batch, N, embed_dim]
        return torch.cat(
            (global_embeddings, cust_embeddings), -2
        )  # [batch, N+1, embed_dim]
