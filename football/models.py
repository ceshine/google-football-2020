import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpClassifierModel(nn.Module):
    def __init__(self, input_dim, hidden_dim: int = 1024, p_dropout: float = 0.25):
        super().__init__()
        self.model = nn.Sequential(
            # nn.BatchNorm1d(input_dim, affine=False),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim, 14)
        )
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight)
                torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                if module.affine:
                    torch.nn.init.constant_(module.weight, 1)
                    module.bias.data.zero_()

    def forward(self, input_tensor):
        return self.model(input_tensor)


class MoeClassifierModel(nn.Module):
    def __init__(self, input_dim, hidden_dim: int = 768, p_dropout: float = 0.2, num_mixtures: int = 4):
        super().__init__()
        self.model = nn.Sequential(
            # nn.BatchNorm1d(input_dim, affine=False),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.num_mixtures = num_mixtures
        self.gating_fc = nn.Sequential(
            # nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim, self.num_mixtures)
        )
        self.expert_fc = nn.Sequential(
            # nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim, 14 * self.num_mixtures)
        )
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight)
                torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                if module.affine:
                    torch.nn.init.constant_(module.weight, 1)
                    module.bias.data.zero_()

    def forward(self, input_tensor):
        fcn_output = self.model(input_tensor)
        expert_logits = self.expert_fc(fcn_output).view(
            -1, 14, self.num_mixtures)
        expert_distributions = F.softmax(
            self.gating_fc(fcn_output), dim=-1
        ).unsqueeze(1)
        logits = (
            expert_logits * expert_distributions  # [..., :self.num_mixtures]
        ).sum(dim=-1)
        return logits
