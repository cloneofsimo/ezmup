import torch
import torch.nn as nn
import torch.nn.functional as F

from ezmup import Ezmup, get_coord_data, plot_coord_data


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, head_dim):
        super(AttentionLayer, self).__init__()
        self.head_dim = head_dim
        # The query, key, and value layers now map from 'hidden_dim' to 'hidden_dim'
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.head_dim**0.5
        attention = F.softmax(attention_scores, dim=-1)
        return torch.matmul(attention, V)


class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, head_dim, num_layers):
        super(MyModel, self).__init__()
        self.layers = nn.ModuleList(
            [AttentionLayer(hidden_dim, head_dim) for _ in range(num_layers)]
        )
        self.fin = nn.Linear(input_dim, hidden_dim)
        self.fout = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fin(x)
        print(x.shape)
        for layer in self.layers:
            x = layer(x)
        x = self.fout(x)
        return x


model = MyModel(
    input_dim=41, output_dim=41, hidden_dim=47 * 2, head_dim=47, num_layers=2
)
model.to("cuda:0")

mup_engine = Ezmup(47, model, init_std=1.0)
mup_engine.change_width_as(64)


def loss_fn(batch, model):
    x, y = batch
    y_pred = model(x)
    return F.mse_loss(y_pred, y)


mup_engine.forward = loss_fn

# example run
x = torch.randn(4, 33, 41).to("cuda:0")
y = torch.randn(4, 33, 41).to("cuda:0")
# y = model(x)


df = get_coord_data(mup_engine, (x, y), n_seeds=1, n_steps=3)
df.to_csv("contents/example.csv")


plot_coord_data(
    df,
    y="l1",
    save_to="contents/coord-check.png",
    suptitle=None,
    x="width",
    hue="module",
)
