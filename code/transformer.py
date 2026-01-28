import math
import torch.nn as nn
import torch

class dot_attention(nn.Module):

    def __init__(self, dim=0):
        super(dot_attention, self).__init__()
        self.dim = dim
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, q, k, v):
        a = torch.tensor(torch.tensor(self.dim),dtype=torch.float)
        head_weight = self.softmax(torch.matmul(q.unsqueeze(-2), k.unsqueeze(-1))/torch.sqrt(a))
        context = torch.matmul(head_weight, v.unsqueeze(-2)).squeeze(-2)

        return context

class multi_head_self_attention_Block(nn.Module):
    def __init__(self, model_dim=100, num_heads=8):
        super(multi_head_self_attention_Block, self).__init__()

        self.dim_per_head = model_dim//num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = dot_attention(self.dim_per_head)
        self.MLP = nn.Sequential(
            nn.Linear(self.dim_per_head * num_heads, self.dim_per_head * num_heads),
            nn.Linear(self.dim_per_head * num_heads, self.dim_per_head * num_heads)
        )


    def forward(self, x, y):
        batch_size = len(x)
        query = self.linear_q(x)
        key = self.linear_k(y)
        value = self.linear_v(y)
        query = query.view(batch_size * self.num_heads, self.dim_per_head)
        key   = key.view(batch_size * self.num_heads, self.dim_per_head)
        value = value.view(batch_size * self.num_heads, self.dim_per_head)
        context = self.dot_product_attention(query, key, value)
        context = context.view(batch_size, self.dim_per_head * self.num_heads)
        x_a = x+context
        output = x_a + self.MLP(x_a)
        return output