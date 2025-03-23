import torch
from .attention_block import MultiHeadEncoder, EmbeddingNet, PositionalEncoding
from torch import nn
import numpy as np

class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            if type(input) == tuple:
                input = module(*input)
            else:
                input = module(input)
        return input


class Feature_Extractor(nn.Module):
    def __init__(self, node_dim = 3, hidden_dim = 16, n_heads = 1, ffh = 16, n_layers = 1,
                 use_positional_encoding = True,
                 is_mlp = False,
                 is_train = False,
                 device = None
                 ):
        super(Feature_Extractor, self).__init__()
        # bs * dim * pop_size * 2 -> bs * dim * pop_size * hidden_dim
        # node_dim = 2 if is_mlp else 3
        self.device = device
        self.embedder = EmbeddingNet(node_dim = node_dim, embedding_dim = hidden_dim)
        self.is_mlp = is_mlp

        self.fes_embedder = EmbeddingNet(node_dim = 1, embedding_dim = 16)
        if not self.is_mlp:
            self.use_PE = use_positional_encoding
            if self.use_PE:
                self.position_encoder = PositionalEncoding(hidden_dim, 512)

            # (bs, dim, pop_size, hidden_dim) -> (bs, dim, pop_size, hidden_dim)
            self.dimension_encoder = mySequential(*(MultiHeadEncoder(n_heads = n_heads,
                                                                     embed_dim = hidden_dim,
                                                                     feed_forward_hidden = ffh,
                                                                     normalization = 'n2')
                                                    for _ in range(n_layers)))
            # (bs, pop_size, dim, hidden_dim) -> (bs, pop_size, dim, hidden_dim)
            self.individual_encoder = mySequential(*(MultiHeadEncoder(n_heads = n_heads,
                                                                      embed_dim = hidden_dim,
                                                                      feed_forward_hidden = ffh,
                                                                      normalization = 'n2')
                                                     for _ in range(n_layers)))
        else:
            self.mlp = nn.Linear(10 + 3, hidden_dim)  # 只是测10D上的问题
            self.acti = nn.ReLU()
        self.is_train = is_train
        self.to(self.device)

    def set_on_train(self):
        self.is_train = True

    def set_off_train(self):
        self.is_train = False

    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.parameters())  # Total number of parameters
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)  # Number of trainable parameters
        return {'Total': total_num, 'Trainable': trainable_num}

    def forward(self, state):
        if self.is_train:
            return self._run(state)
        else:
            with torch.no_grad():
                return self._run(state)

    def encode_y(self, ys, constant = 32.0):
        """
            Encodes the input tensor `ys` by separating it into mantissa and exponent parts.

            Parameters:
            - ys: Input numpy_array of shape (bs, n), where bs is the batch size and n is the population size.
            - constant: A scaling constant for the exponent part, default is 32.0.

            Returns:
            - encoded_y: The mantissa part of the encoded `ys`, scaled by 1/10, shape (bs, n).
            - encoded_e: The exponent part of the encoded `ys`, scaled by the given constant, shape (bs, n).
        """
        bs, n = ys.shape
        mask = ys != 0
        exponent = np.floor(np.log10(np.abs(ys[mask])))
        mantissa = ys[mask] / (10 ** exponent)

        encoded_y = mantissa.reshape(bs, n) / 10
        encoded_e = exponent.reshape(bs, n) + 1
        encoded_e = encoded_e / constant

        return encoded_y, encoded_e

    def _run(self, state):
        """
            Processes the input tensors and applies attention mechanisms based on the selected order.

            Parameters:
            - xs: Input numpy_array of shape (bs, n, d),
                where bs is the batch size, n is the population size, and d is the dimension.
            - ys: Input numpy_array of shape (bs, n), representing target values.

            Returns:
            - out: Processed output tensor, depending on the attention mechanism used,
                with shape (bs, pop_size, hidden_dim).
        """
        # xs : bs * n * d
        # ys : bs * n
        # fes : bs * 1
        xs, ys, fes = state['x'], state['y'], state['fes']
        bs, pop_size, dim = xs.shape

        y_, e_ = self.encode_y(ys)  # 2D : bs * n and bs * n
        ys = y_[:, :, None]  # 3D : bs * n * 1
        es = e_[:, :, None]  # 3D : bs * n * 1

        a_x = xs[:, :, :, None]  # 4D : bs * n * d * 1
        a_y = np.repeat(ys, dim, -1)[:, :, :, None]  # 4D : bs * n * d * 1
        a_e = np.repeat(es, dim, -1)[:, :, :, None]  # 4D : bs * n * d * 1

        # a_fes = np.repeat(fes, pop_size, -1)[:, :, None] # 3D : bs * n * 1
        # a_fes = np.repeat(a_fes, dim, -1)[:, :, :, None] # 4D : bs * n * d * 1

        raw_feature = np.concatenate([a_x, a_y, a_e], axis = -1).transpose((0, 2, 1, 3))  # 4D : bs * d * n * 3

        h_ij = self.embedder(torch.tensor(raw_feature, dtype = torch.float32).to(self.device))  # 4D : bs * d * n * hidden_dim

        node_dim = h_ij.shape[-1]

        h_ij = h_ij.view(-1, pop_size, node_dim)  # resize h_ij 3D : (bs * dim, pop_size, hidden_dim)

        if not self.is_mlp:
            # (bs * dim, pop_size, hidden_dim) -> (bs, dim, pop_size, hidden_dim)
            o_ij = self.dimension_encoder(h_ij).view(bs, dim, pop_size, node_dim).to(self.device)
            # (bs, pop_size, dim, hidden_dim) -> (bs * pop_size, dim, hidden_dim)
            o_i = o_ij.permute(0, 2, 1, 3).contiguous().view(-1, dim, node_dim).to(self.device)

            if self.use_PE:
                o_i = o_i + self.position_encoder.get_PE(dim).to(self.device) * 0.5

            # mean
            # o_i = torch.mean(o_i, 1).view(bs, pop_size, node_dim).to(self.device) # (bs, pop_size, hidden_dim)

            tensor_fes = torch.tensor(fes, dtype = torch.float32).to(self.device)
            embed_fes = self.fes_embedder(tensor_fes).to(self.device)  # [bs, 16]
            embed_fes = embed_fes.unsqueeze(1)  # [bs, 1, 16]
            embed_fes = embed_fes.expand(-1, pop_size, -1)  # [bs, pop_size, 16]

            # o_i = self.individual_encoder(o_i).to(self.device) # (bs, pop_size, hidden_dim)
            o_i = self.individual_encoder(o_i).view(bs, pop_size, dim, node_dim).to(self.device)

            o_i = torch.mean(o_i, 2).to(self.device)
            out = torch.cat((o_i, embed_fes), -1).to(self.device)  # (bs, pop_size, hidden_dim + 16)

        else:
            # out = self.acti(self.mlp(h_ij)).view(bs, dim, pop_size, node_dim)
            a_x = xs[:, :, :]  # 3D: bs * n * d
            a_y = ys  # 3D: bs * n * 1
            a_e = es  # 3D: bs * n * 1
            a_fes = np.repeat(fes, pop_size, -1)[:, :, None]  # 3D : bs * n * 1

            mlp_feature = np.concatenate([a_x, a_y, a_e, a_fes], axis = -1)

            h_ij = torch.tensor(mlp_feature, dtype = torch.float32).to(self.device)

            # out = self.mlp(mlp_feature).view(bs, dim, pop_size, node_dim)
            out = self.mlp(h_ij)  # 3D : bs * pop_size * hidden_dim
        return out



if __name__ == "__main__":
    from attention_block import MultiHeadEncoder, EmbeddingNet, PositionalEncoding
    FE = Feature_Extractor(is_train = True)
    x = np.random.rand(4, 5, 2)
    y = np.random.rand(4, 5)

    feature = FE(x, y)
    print(feature.shape)
