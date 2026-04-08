import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal

__all__ = ["Task"]
class ParallelGatedMLP(nn.Module):
    """From https://github.com/togethercomputer/stripedhyena"""

    def __init__(
        self,
        hidden_size: int = 128,
        inner_size_multiple_of: int = 256,
        mlp_activation: str = "silu",
        model_parallel_size: int = 1,
    ):
        super().__init__()

        multiple_of = inner_size_multiple_of
        self.act_type = mlp_activation
        if self.act_type == "gelu":
            self.act = F.gelu
        elif self.act_type == "silu":
            self.act = F.silu
        else:
            raise NotImplementedError

        self.multiple_of = multiple_of * model_parallel_size

        inner_size = int(2 * hidden_size * 4 / 3)
        inner_size = self.multiple_of * (
            (inner_size + self.multiple_of - 1) // self.multiple_of
        )

        self.l1 = nn.Linear(
            in_features=hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l2 = nn.Linear(
            in_features=hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l3 = nn.Linear(
            in_features=inner_size,
            out_features=hidden_size,
            bias=False,
        )

    def forward(self, z):
        z1, z2 = self.l1(z), self.l2(z)
        return self.l3(self.act(z1) * z2)

class MLP(nn.Module):
    def __init__(self, input_size=128, output_size=128, hidden_size=512):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class Task(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            num_experts,
            hidden_size=None,
            k=1,
            T=1.0,
            noisy_gating=True,
            routing_level="node",
            routing_method="input_choice",
            CoE_model="MLP",
            sign=False
    ):
        super(Task, self).__init__()
        self.noisy_gating = noisy_gating
        self.routing_level = routing_level
        self.routing_method = routing_method
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        self.T = T
        self.sign = sign
        if CoE_model == "MLP":
            self.experts = nn.ModuleList(
                [
                    MLP(self.input_size, self.output_size, self.hidden_size)
                    for _ in range(self.num_experts)
                ]
            )
        elif CoE_model == "RF":
            self.experts = nn.ModuleList(
                [
                    ParallelGatedMLP()
                    for _ in range(self.num_experts)
                ]
            )
        elif CoE_model == "Linear":
            self.experts = nn.ModuleList(
                [
                    nn.Linear(self.input_size, self.output_size, bias=False)
                    for _ in range(self.num_experts)
                ]
            )
        else:
            raise NotImplementedError

        self.shared_gate = SharedGate(input_size, 16)
        self.task_gates = TaskSpecificGate(input_size, num_experts, self.noisy_gating, self.T)


        self.softplus = nn.Softplus()

        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert (self.k <= self.num_experts)

    def forward(self, x, active_experts, loss_coef=1e-2):
        assert self.routing_level in ["node", "instance", "problem"], "Unsupported Routing Level!"
        output_shape = list(x.size()[:-1]) + [
            self.output_size]
        if self.routing_level in ["instance", "problem"]:
            assert x.dim() == 3
        elif self.routing_level == "node":
            x = x.reshape(-1, self.input_size) if x.dim() != 2 else x
        shared_scores = self.shared_gate(x) + x
        specific_scores = self.task_gates(shared_scores, self.training, active_experts)


        outputs = []
        out_expert_all = []
        ind = 0
        for i, expert in enumerate(self.experts):
            out = expert(x)
            out_expert_all.append(out)
            ind += 1
        aux_loss = 0
        num_experts = len(out_expert_all)

        batch_size, num_nodes, _ = output_shape
        out_expert_stack = torch.stack(out_expert_all, dim=1) 
        out_reshaped = out_expert_stack.view(batch_size, num_nodes, active_experts.size()[-1], 128)
        weights = specific_scores.unsqueeze(-1)


        output = out_reshaped * weights
        output_end = torch.sum(output, dim=2)

        if self.sign:
            B, N, _ = output_shape


            mask = active_experts.view(B, 1, self.num_experts, 1).float()

            count = mask.sum(dim=2)

            avg_expert = (out_reshaped * mask).sum(dim=2) / count

            mse_per_expert = ((out_reshaped - avg_expert.unsqueeze(2)) ** 2).mean(dim=3)

            mse_per_expert = mse_per_expert * mask.squeeze(-1)

            loss_node = mse_per_expert.sum(dim=2) / count.squeeze(-1)

            is_two = (count.squeeze(-1) == 2)
            loss_node = torch.where(is_two, loss_node * 4, loss_node)

            loss_node = torch.where(count.squeeze(-1) < 2, torch.zeros_like(loss_node), loss_node)

            aux_loss = loss_node.mean() * loss_coef 

        return output_end, aux_loss




class SharedGate(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SharedGate, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class TaskSpecificGate(nn.Module):
    def __init__(self, input_dim, num_experts, noisy_gating, T=1.0):
        super(TaskSpecificGate, self).__init__()
        self.noisy_gating = noisy_gating
        self.T = T
        
        self.gates = nn.ModuleList([nn.Linear(input_dim, 1, bias=False) for _ in range(num_experts)])

    def forward(self, x, train, active_experts, noise_epsilon=1e-2):
        scores = []
        for i, gate in enumerate(self.gates):
            scores.append(gate(x))


        scores = torch.cat(scores, dim=-1)
        s_reshape = scores.reshape(active_experts.size()[0], -1, active_experts.size()[-1])
        mask = active_experts.bool().unsqueeze(1)
        x_masked = s_reshape.masked_fill(~mask, float('-inf'))
        scores_end = torch.softmax(x_masked, dim=-1)

        return scores_end
