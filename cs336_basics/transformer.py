import torch
from einops import einsum, rearrange, repeat
from cs336_basics.nn_utils import softmax

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        '''
        Construct a linear transformation module.
        
        Args:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(out_features, in_features, dtype=dtype, device=device))
        # Xavier normal initialization with 3 std truncation
        std = (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weights, 0, std, -3 * std, 3 * std)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        """
        return x@self.weights.T
    

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        '''
        Construct an embedding module.
        
        Args:
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(num_embeddings, embedding_dim, dtype=dtype, device=device))
        # Normal initialization with 3 std truncation
        torch.nn.init.trunc_normal_(self.weights, 0, 1, -3, 3)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        """
        return self.weights[token_ids]
    
    
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        '''
        Construct the RMSNorm module.
        
        Args:
            d_model: int Hidden dimension of the model
            eps: float = 1e-5 Epsilon value for numerical stability
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(d_model, dtype=dtype, device=device))
        self.eps = eps
        # Normal initialization
        torch.nn.init.normal_(self.weights)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        """
        in_dtype = x.dtype
        # upcast to float32 to prevent overflow
        x = x.to(torch.float32)
        # Perform RMSNorm
        rms = torch.rsqrt(einsum(x, x, 'b s d, b s d -> b s')/self.weights.shape[0]+self.eps)
        rms = rearrange(rms, 'b s -> b s 1')
        result = x * rms * self.weights
        return result.to(in_dtype)
    

def silu(x: torch.Tensor) -> torch.Tensor:
    """
    Apply SiLU activation function
    """
    return x * torch.sigmoid(x)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        '''
        Construct the SwiGLU module.
        
        Args:
            d_model: int Hidden dimension of the model
            d_ff: int dimension of inner feed-forward layer
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.randn(d_ff, d_model, dtype=dtype, device=device))
        self.w2 = torch.nn.Parameter(torch.randn(d_model, d_ff, dtype=dtype, device=device))
        self.w3 = torch.nn.Parameter(torch.randn(d_ff, d_model, dtype=dtype, device=device))
        std = (2 / (d_model + d_ff)) ** 0.5
        torch.nn.init.trunc_normal_(self.w1, 0, std, -3 * std, 3 * std)
        torch.nn.init.trunc_normal_(self.w2, 0, std, -3 * std, 3 * std)
        torch.nn.init.trunc_normal_(self.w3, 0, std, -3 * std, 3 * std)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the SwiGLU transformation to the input.
        """
        x1 = silu(x@self.w1.T)
        x3 = x@self.w3.T
        return (x1 * x3)@self.w2.T
    
    
class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        '''
        Construct the RoPE module and create buffers if needed.
        
        Args:
            theta: float Θ value for the RoPE
            d_k: int dimension of query and key vectors
            max_seq_len: int Maximum sequence length that will be inputted
            device: torch.device | None = None Device to store the buffer on
            dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        theta_vec = 1.0 / (theta) ** (torch.arange(0, d_k, 2, device=device, dtype=dtype)/d_k)
        pos = torch.arange(0, max_seq_len, 1, device=device, dtype=dtype)
        freqs = torch.outer(pos, theta_vec)
        self.register_buffer('cos', freqs.cos(), persistent=False)
        self.register_buffer('sin', freqs.sin(), persistent=False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor=None) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        """
        # repeat cos and sin
        cos = self.cos[token_positions] if token_positions is not None else self.cos[:x.shape[-2]]
        cos = repeat(cos, '... d -> ... (d n)', n=2)
        sin = self.sin[token_positions] if token_positions is not None else self.sin[:x.shape[-2]]
        sin = repeat(sin, '... d -> ... (d n)', n=2)
        # rotate x = [-x2 x1 -x4, x3, ...]
        stacked_x = rearrange(x, '... (d n) -> ... d n', n=2)
        rotated_x = rearrange([-stacked_x[..., 1], stacked_x[..., 0]], 'n ... d -> ... (d n)', n=2)
        # x1*cos-x2*sin, x2*cos+x1*sin, ...
        return x * cos + rotated_x * sin


def scale_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor=None):
    qk = einsum(q, k, '... n d_k, ... m d_k -> ... n m') / q.shape[-1] ** 0.5
    if mask is not None:
        qk = qk.masked_fill(~mask, float('-inf'))
    qkv = einsum(softmax(qk, -1), v, 'b ... n m, b ... m d_v -> b ... n d_v')
    return qkv


class CausalMultiHeadedSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float=0, max_seq_len: int=-1, device=None, dtype=None):
        '''
        Construct the Causal MHA module.
        
        Args:
            d_model: int Dimensionality of the Transformer block inputs.
            num_heads: int Number of heads to use in multi-head self-attention.
            theta: float Θ value for the RoPE
            max_seq_len: int Maximum sequence length that will be inputted
            device: torch.device | None = None Device to store the buffer on
            dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.num_heads = num_heads
        self.w_q = torch.nn.Parameter(torch.randn(d_model, d_model, dtype=dtype, device=device)) # h d_k * d_model
        self.w_k = torch.nn.Parameter(torch.randn(d_model, d_model, dtype=dtype, device=device)) # h d_k * d_model
        self.w_v = torch.nn.Parameter(torch.randn(d_model, d_model, dtype=dtype, device=device)) # h d_v * d_model
        self.w_o = torch.nn.Parameter(torch.randn(d_model, d_model, dtype=dtype, device=device)) # d_model * h d_v
        std = (1 / d_model) ** 0.5
        torch.nn.init.trunc_normal_(self.w_q, 0, std, -3 * std, 3 * std)
        torch.nn.init.trunc_normal_(self.w_k, 0, std, -3 * std, 3 * std)
        torch.nn.init.trunc_normal_(self.w_v, 0, std, -3 * std, 3 * std)
        torch.nn.init.trunc_normal_(self.w_o, 0, std, -3 * std, 3 * std)
        self.rope = RotaryPositionalEmbedding(theta, d_model/num_heads, max_seq_len, device, dtype) if max_seq_len > 0 else None
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor=None) -> torch.Tensor:
        """
        Apply batched multi-headed attention.
        """
        # Compute Q, K, V
        x_q = rearrange(einsum(x, self.w_q, '... s d, hd d -> ... s hd'), '... s (h d_k) -> ... h s d_k', h=self.num_heads)
        x_k = rearrange(einsum(x, self.w_k, '... s d, hd d -> ... s hd'), '... s (h d_k) -> ... h s d_k', h=self.num_heads)
        x_v = rearrange(einsum(x, self.w_v, '... s d, hd d -> ... s hd'), '... s (h d_v) -> ... h s d_v', h=self.num_heads)
        # TODO: Combine the key, query, and value projections into a single weight matrix for a single matrix multiply
        
        # Apply RoPE to Q and K only
        if self.rope:
            x_q = self.rope(x_q, token_positions)
            x_k = self.rope(x_k, token_positions)
        
        # Compute SDPA
        n = x.shape[-2]
        mask = torch.tril(torch.ones(n, n, dtype=torch.bool, device=x.device))
        mha = rearrange(scale_dot_product_attention(x_q, x_k, x_v, mask), '... h s d_v -> ... s (h d_v)')
        
        # Compute output
        return einsum(mha, self.w_o, '... s hd, d hd -> ... s d')


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float=0, max_seq_len: int=-1, device=None, dtype=None):
        '''
        Construct the Transforomer block module.
        
        Args:
            d_model: int Dimensionality of the Transformer block inputs.
            num_heads: int Number of heads to use in multi-head self-attention.
            d_ff: Dimensionality of the position-wise feed-forward inner layer.
            theta: float Θ value for the RoPE
            max_seq_len: int Maximum sequence length that will be inputted
            device: torch.device | None = None Device to store the buffer on
            dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model,
                           device=device,
                           dtype=dtype)
        self.attn = CausalMultiHeadedSelfAttention(d_model=d_model,
                                                   num_heads=num_heads,
                                                   theta=theta,
                                                   max_seq_len=max_seq_len,
                                                   device=device,
                                                   dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model,
                           device=device,
                           dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model,
                          d_ff=d_ff,
                          device=device,
                          dtype=dtype)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor=None) -> torch.Tensor:
        """
        Apply the tranformer block.
        """
        # Multi-headed Attention
        y = x + self.attn(self.ln1(x), token_positions)
        # Feed-forward
        return y + self.ffn(self.ln2(y))


class Transformer(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length:int, d_model: int, num_layers: int,
                 num_heads: int, d_ff: int, rope_theta: float=0,
                 device=None, dtype=None):
        '''
        Construct the Transforomer module.
        
        Args:
            vocab_size: int The size of the vocabulary, necessary for determining the dimensionality of the token embedding matrix.
            context_length: int The maximum context length, necessary for determining the dimensionality of the position embedding matrix.
            d_model: int Dimensionality of the Transformer block inputs.
            num_layers: int The number of Transformer blocks to use.
            num_heads: int Number of heads to use in multi-head self-attention.
            d_ff: Dimensionality of the position-wise feed-forward inner layer.
            rope_theta: float Θ value for the RoPE
            device: torch.device | None = None Device to store the buffer on
            dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size,
                                   embedding_dim=d_model,
                                   device=device,
                                   dtype=dtype)
        self.attn_layers = torch.nn.ModuleList([TransformerBlock(d_model=d_model,
                                              num_heads=num_heads,
                                              d_ff=d_ff,
                                              theta=rope_theta,
                                              max_seq_len=context_length,
                                              device=device,
                                              dtype=dtype)
                             for i in range(num_layers)])
        self.ln = RMSNorm(d_model=d_model,
                          device=device,
                          dtype=dtype)
        self.out = Linear(in_features=d_model,
                          out_features=vocab_size,
                          device=device,
                          dtype=dtype)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply the tranformer.
        """
        x = self.embedding(token_ids)
        positions = torch.arange(token_ids.shape[-1], device=token_ids.device)
        for layer in self.attn_layers:
            x = layer(x, positions)
        x = self.ln(x)
        x = self.out(x)
        return x
