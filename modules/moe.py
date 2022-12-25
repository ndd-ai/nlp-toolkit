import torch


class SoftmaxGate(torch.nn.Module):
    def __init__(self, embed_dim, num_experts, intermediate_dim=128):
        super().__init__()
        self.linear1 = torch.nn.Linear(
            in_features=embed_dim, out_features=intermediate_dim
        )
        self.linear2 = torch.nn.Linear(
            in_features=intermediate_dim, out_features=num_experts
        )
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, embed):
        intermediate_vec = self.linear1(embed)
        logits = self.linear2(intermediate_vec)
        probs = self.sm(logits)
        return probs
