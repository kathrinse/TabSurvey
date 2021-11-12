import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from models.basemodel import BaseModel
from utils.io_utils import get_output_path


class TabTransformer(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        num_continuous = args.num_features  # Todo: Adapt this for cat data!
        categories_unique = ()  # Todo: Adapt this for cat data!

        if args.cat_idx:
            self.num_idx = list(set(range(args.num_features)) - set(args.cat_idx))
        else:
            self.num_idx = list(range(args.num_features))

        self.device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
        print("On Device:", self.device)

        self.model = TabTransformerModel(
            categories=categories_unique,  # tuple containing the number of unique values within each category
            num_continuous=num_continuous,  # number of continuous values
            dim=32,  # dimension, paper set at 32
            dim_out=args.num_classes,
            depth=6,  # depth, paper recommended 6
            heads=8,  # heads, paper recommends 8
            attn_dropout=0.1,  # post-attention dropout
            ff_dropout=0.1,  # feed forward dropout
            mlp_hidden_mults=(4, 2),  # relative multiples of each hidden dimension of the last mlp to logits
            mlp_act=nn.ReLU(),  # activation for final mlp, defaults to relu, but could be anything else (selu etc)
        ).to(self.device)

    def fit(self, X, y, X_val=None, y_val=None):
        optimizer = optim.AdamW(self.model.parameters(), lr=0.01)

        X = torch.tensor(X).float()
        X_val = torch.tensor(X_val).float()

        y = torch.tensor(y)
        y_val = torch.tensor(y_val)

        if self.args.objective == "regression":
            loss_func = nn.MSELoss()
            y = y.float()
            y_val = y_val.float()
        elif self.args.objective == "classification":
            loss_func = nn.CrossEntropyLoss()

        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True,
                                  num_workers=2)

        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(dataset=val_dataset, batch_size=128, shuffle=True)

        val_dim = y_val.shape[0]
        min_val_loss = float("inf")
        min_val_loss_idx = 0

        for epoch in range(self.args.epochs):
            for i, (batch_X, batch_y) in enumerate(train_loader):

                if self.args.cat_idx:
                    x_categ = batch_X[:, self.args.cat_idx].to(self.device)
                else:
                    x_categ = None
                x_cont = batch_X[:, self.num_idx].to(self.device)

                out = self.model(x_categ, x_cont)

                if self.args.objective == "regression":
                    out = out.squeeze()

                loss = loss_func(out, batch_y.to(self.device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Early Stopping
                val_loss = 0.0
                for val_i, (batch_val_X, batch_val_y) in enumerate(val_loader):

                    if self.args.cat_idx:
                        x_categ = batch_val_X[:, self.args.cat_idx].to(self.device)
                    else:
                        x_categ = None

                    x_cont = batch_val_X[:, self.num_idx].to(self.device)

                    out = self.model(x_categ, x_cont)

                    if self.args.objective == "regression":
                        out = out.squeeze()

                    val_loss += loss_func(out, batch_val_y.to(self.device))
                val_loss /= val_dim

                current_idx = (i + 1) * (epoch + 1)

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    min_val_loss_idx = current_idx

                if min_val_loss_idx + self.args.early_stopping_rounds < current_idx:
                    # print("Validation loss has not improved for %d steps!" % self.args.early_stopping_rounds)
                    # print("Early stopping applies.")
                    return

    def predict(self, X):
        X = torch.tensor(X).float()

        if self.args.cat_idx:
            x_categ = X[:, self.args.cat_idx].to(self.device)
        else:
            x_categ = None

        x_cont = X[:, self.num_idx].to(self.device)
        self.predictions = self.model(x_categ, x_cont).detach().numpy()

        return self.predictions

    def save_model(self, filename_extension=""):
        filename = get_output_path(self.args, directory="models", filename="m", extension=filename_extension,
                                   file_type="pt")
        torch.save(self.model.state_dict(), filename)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        return dict()


####################################################################################################################
#
#  TabTransformer code from
#  https://github.com/lucidrains/tab-transformer-pytorch/blob/main/tab_transformer_pytorch/tab_transformer_pytorch.py
#  adapted to work without categorical data
#
#####################################################################################################################
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange


# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


# transformer

class Transformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.embeds = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                Residual(PreNorm(dim, FeedForward(dim, dropout=ff_dropout))),
            ]))

    def forward(self, x):
        x = self.embeds(x)

        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        return x


# mlp

class MLP(nn.Module):
    def __init__(self, dims, act=None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims_pairs) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                self.dim_out = dim_out
                continue

            act = default(act, nn.ReLU())
            layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlp(x)

        # Added for multiclass output!
        if self.dim_out > 1:
            x = torch.softmax(x, dim=1)
        return x


# main class

class TabTransformerModel(nn.Module):
    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            depth,
            heads,
            dim_head=16,
            dim_out=1,
            mlp_hidden_mults=(4, 2),
            mlp_act=None,
            num_special_tokens=2,
            continuous_mean_std=None,
            attn_dropout=0.,
            ff_dropout=0.
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]
        self.register_buffer('categories_offset', categories_offset)

        # continuous

        if exists(continuous_mean_std):
            assert continuous_mean_std.shape == (num_continuous,
                                                 2), f'continuous_mean_std must have a shape of ({num_continuous}, 2) where the last dimension contains the mean and variance respectively'
        self.register_buffer('continuous_mean_std', continuous_mean_std)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous

        # transformer

        self.transformer = Transformer(
            num_tokens=total_tokens,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )

        # mlp to logits

        input_size = (dim * self.num_categories) + num_continuous
        l = input_size // 8

        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act=mlp_act)

    def forward(self, x_categ, x_cont):

        # Adaption to work without categorical data
        if x_categ is not None:
            assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} ' \
                                                             f'values for your categories input'
            x_categ += self.categories_offset
            x = self.transformer(x_categ)
            flat_categ = x.flatten(1)

        assert x_cont.shape[1] == self.num_continuous, f'you must pass in {self.num_continuous} ' \
                                                       f'values for your continuous input'

        if exists(self.continuous_mean_std):
            mean, std = self.continuous_mean_std.unbind(dim=-1)
            x_cont = (x_cont - mean) / std

        normed_cont = self.norm(x_cont)

        # Adaption to work without categorical data
        if x_categ is not None:
            x = torch.cat((flat_categ, normed_cont), dim=-1)
        else:
            x = normed_cont

        return self.mlp(x)
