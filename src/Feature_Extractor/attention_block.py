import torch
import torch.nn.functional as F
from torch import nn
import math

# USE_PE? just add some bias

# implements Normalization module
class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalization = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.normalization(x)

class PositionalEncoding():
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def get_PE(self, seq_len):
        return self.encoding[:seq_len, :]

# implements Multi-head Self-Attemtion module
class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, input_dim, embed_dim=None, val_dim=None, key_dim=None):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)

        self.W_q = nn.Parameter(torch.randn(n_heads, input_dim, key_dim))
        self.W_k = nn.Parameter(torch.randn(n_heads, input_dim, key_dim))
        self.W_v = nn.Parameter(torch.randn(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_o = nn.Parameter(torch.randn(n_heads, val_dim, embed_dim))

    def forward(self, q, h=None):
        # MultiHead(Q,K,V) = Concat(head1, ... ,headh)W
        # where headi = Attention(QW_q,KW_k,Vw_V)

        # q : bs n_q, input_dim
        # h : bs n_h, input_dim

        if h is None:
            h = q # self-attention
        bs, n_h, input_dim = h.size()
        n_q = q.size(1)
        hflat = h.contiguous().view(-1, input_dim) # 2D : bs * n_h, input _dim
        qflat = q.contiguous().view(-1, input_dim) # 2D : bs * n_q, input_dim

        shape_h = (self.n_heads, bs, n_h, -1)
        shape_q = (self.n_heads, bs, n_q, -1)

        # Calculate queries
        Q = torch.matmul(qflat, self.W_q).view(shape_q) # 2D * 3D (bs * n_q, input_dim) * (n_heads, input_dim, key_dim)
        # -> 3D (n_heads, bs * n_q, key_dim) -> (n_heads, bs, n_q, key_dim)

        # 4D : (n_heads, bs, n_h, key_dim) or (n_heads, bs, n_h, val_dim)
        K = torch.matmul(hflat, self.W_k).view(shape_h)
        V = torch.matmul(hflat, self.W_v).view(shape_h)

        # Calculate matmul(Q, K) 4D : (n_heads, bs, n_q, n_h)
        attention_scores = self.norm_factor * torch.matmul(Q, K.transpose(-1, -2))
        attn = F.softmax(attention_scores, dim = -1)

        # 4D : (n_heads, bs, n_q, val_dim)
        heads = torch.matmul(attn, V)
        out = torch.mm(
            # (bs, n_q, n_heads, val_dim) -> (bs * n_q, n_heads * val_dim)
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            # (n_heads * val_size, embed_dim)
            self.W_o.view(-1, self.embed_dim)
        ).view(bs, n_q, self.embed_dim)

        # 3D : (bs, n_q, embed_dim)
        return out

# implements (DAC-Att sublayer)
class MultiHeadAttentionsubLayer(nn.Module):
    def __init__(self, n_heads, embed_dim, normalization):
        super(MultiHeadAttentionsubLayer, self).__init__()
        self.MHA = MultiHeadAttention(n_heads, input_dim = embed_dim, embed_dim = embed_dim)
        self.Norm = Normalization(embed_dim)
    def forward(self, x):
        # Get Attention
        out = self.MHA(x)
        return self.Norm(out + x)

# implements the encoder (FFN sublayer)
class FFandNormsubLayer(nn.Module):
    def __init__(self, embed_dim, feed_forward_hidden,normalization):
        super(FFandNormsubLayer, self).__init__()
        self.FF = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_hidden, embed_dim)
        ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)

        self.Norm = Normalization(embed_dim)

    def forward(self, x):
        # Get FF
        out = self.FF(x)
        return self.Norm(out + x)

# implements MultiHeadEncoder
class MultiHeadEncoder(nn.Module):
    def __init__(self, n_heads, embed_dim, feed_forward_hidden, normalization):
        super(MultiHeadEncoder, self).__init__()
        self.MHA_sublayer = MultiHeadAttentionsubLayer(n_heads, embed_dim, normalization = normalization)
        self.FFandNorm_sublayer = FFandNormsubLayer(embed_dim, feed_forward_hidden, normalization = normalization)
    def forward(self, x):
        out = self.MHA_sublayer(x)
        return self.FFandNorm_sublayer(out)

class EmbeddingNet(nn.Module):

    def __init__(self,
                 node_dim,
                 embedding_dim):
        super(EmbeddingNet, self).__init__()
        self.node_dim = node_dim
        self.embedding_dim = embedding_dim
        self.embedder = nn.Linear(node_dim, embedding_dim, bias=False)

    def forward(self, x):
        h_em = self.embedder(x)
        return h_em