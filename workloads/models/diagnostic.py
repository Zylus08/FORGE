"""Diagnostic microbenchmarks for FORGE M1."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComputeBoundMLP(nn.Module):
    """4-layer MLP for compute-bound GEMM evaluation (batch=1024)."""
    def __init__(self, in_features=1024, hidden_dim=2048, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class MemoryBoundOperatorChain(nn.Module):
    """Memory-bound embedding and elementwise arithmetic chain."""
    def __init__(self, num_embeddings=10000, embedding_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.bias = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, indices):
        # Memory lookup
        x = self.embedding(indices)
        # Elementwise operations: low arithmetic intensity
        x = x * self.scale + self.bias
        x = F.relu(x)
        x = x + 0.5 * torch.tanh(x)
        return x
