"""
TRANSFORMER FROM SCRATCH  ("Attention Is All You Need", Vaswani et al. 2017)
=============================================================================
Every module below is written out by hand — no nn.MultiheadAttention,
no nn.Transformer. Only nn.Linear, nn.Embedding, nn.LayerNorm (a normalization
formula, not "the attention trick") and raw tensor ops are used, so you can
see exactly where every equation from the paper lives in code.

Notation used throughout (matches the paper):
    B  = batch size
    T  = sequence length (source or target)
    d_model = embedding / residual-stream width      (paper: 512)
    h  = number of attention heads                    (paper: 8)
    d_k = d_v = d_model / h = per-head dimension       (paper: 64)
    d_ff = inner feed-forward width                    (paper: 2048)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. SCALED DOT-PRODUCT ATTENTION
# =============================================================================
#
#   Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
#
# Intuition:
#   - Q (query)  : "what am I looking for at this position?"
#   - K (key)    : "what do I contain, that others might look for?"
#   - V (value)  : "what do I actually give you, if you attend to me?"
#   - Q K^T      : dot product similarity between every query and every key
#                  -> shape (T_q, T_k), a matrix of raw attention scores
#   - / sqrt(d_k): scaling. Without it, dot products grow with d_k in
#                  magnitude, which pushes softmax into regions with tiny
#                  gradients (it saturates). Dividing by sqrt(d_k) keeps the
#                  variance of the scores ~1 regardless of dimension.
#   - softmax(.) : turns scores into a probability distribution per query
#                  row (each row sums to 1) -> "how much should I mix in
#                  each value vector"
#   - (.) V      : weighted sum of value vectors using those probabilities
#
# The optional `mask` sets disallowed positions to -inf BEFORE the softmax,
# so they receive 0 probability after it. Two uses in this file:
#   - causal mask   (decoder self-attn: position t cannot see t+1, t+2, ...)
#   - padding mask   (ignore <pad> tokens)
def scaled_dot_product_attention(q, k, v, mask=None):
    # q: (B, h, T_q, d_k)   k: (B, h, T_k, d_k)   v: (B, h, T_k, d_v)
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)   # (B, h, T_q, T_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)                    # normalize over T_k
    out = attn @ v                                      # (B, h, T_q, d_v)
    return out, attn


# =============================================================================
# 2. MULTI-HEAD ATTENTION
# =============================================================================
#
#   MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W^O
#   head_i           = Attention(Q W_i^Q, K W_i^K, V W_i^V)
#
# Instead of doing attention once with the full d_model-dim vectors, the
# paper projects Q,K,V into h smaller subspaces (d_k = d_model/h each) and
# runs attention independently in each subspace. Why bother?
#   - A single attention head is forced to average over ONE similarity
#     pattern per position. Multiple heads let the model attend to
#     different *kinds* of relationships simultaneously (e.g. one head
#     tracks "the previous noun", another tracks "matching verb tense").
#   - Because each head is smaller (d_k = d_model/h), the total compute is
#     about the same as one big head, but representational flexibility
#     goes up.
# W^Q, W^K, W^V, W^O are all learned linear layers.
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        assert d_model % h == 0
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        # In practice we don't loop over h separate small Linear layers;
        # one big Linear(d_model, d_model) computes all heads' projections
        # at once, and we just *reshape* the output into h heads.
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)   # final output projection W^O

    def split_heads(self, x):
        # (B, T, d_model) -> (B, h, T, d_k)
        B, T, _ = x.shape
        x = x.view(B, T, self.h, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        # (B, h, T, d_k) -> (B, T, d_model)
        B, h, T, d_k = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, T, h * d_k)

    def forward(self, q_in, k_in, v_in, mask=None):
        # q_in/k_in/v_in are the SAME tensor for self-attention, but
        # different tensors for encoder-decoder cross-attention
        # (q_in = decoder states, k_in = v_in = encoder outputs).
        q = self.split_heads(self.w_q(q_in))   # (B, h, T_q, d_k)
        k = self.split_heads(self.w_k(k_in))   # (B, h, T_k, d_k)
        v = self.split_heads(self.w_v(v_in))   # (B, h, T_k, d_k)

        out, attn = scaled_dot_product_attention(q, k, v, mask)
        out = self.combine_heads(out)          # (B, T_q, d_model)
        return self.w_o(out), attn


# =============================================================================
# 3. POSITION-WISE FEED-FORWARD NETWORK
# =============================================================================
#
#   FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
#
# Attention mixes information ACROSS positions/tokens. It has no nonlinearity
# of its own beyond softmax, and it can't do per-position feature
# transformation. The FFN is applied to each position independently and
# identically (same weights at every position) - this is where most of the
# model's per-token "thinking" / nonlinear feature recombination happens.
# d_ff is much wider than d_model (2048 vs 512 in the paper) - expand, apply
# ReLU, project back down.
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(F.relu(self.w_1(x)))


# =============================================================================
# 4. POSITIONAL ENCODING
# =============================================================================
#
#   PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
#   PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )
#
# Self-attention is PERMUTATION INVARIANT: Attention(Q,K,V) gives the exact
# same output if you shuffle the token order (only the row/col labels
# change). So the model has literally no idea what order the tokens came
# in unless we inject that information. The paper adds a fixed
# (non-learned) vector to each token embedding that encodes its position,
# using sinusoids of geometrically increasing wavelengths (from 2*pi to
# 10000*2*pi). Two nice properties:
#   - Each dimension is a different frequency sine/cosine wave, so nearby
#     positions get similar encodings and far ones don't (smooth, unique).
#   - PE(pos+k) is a LINEAR function of PE(pos) (a rotation), which the
#     authors hoped would let the model learn to attend by relative
#     position easily.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()          # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                              * (-math.log(10000.0) / d_model))           # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)   # even dims -> sin
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dims  -> cos
        self.register_buffer('pe', pe.unsqueeze(0))    # (1, max_len, d_model), not a learned param

    def forward(self, x):
        # x: (B, T, d_model) token embeddings -> add positional info
        T = x.size(1)
        return x + self.pe[:, :T]


# =============================================================================
# 5. RESIDUAL CONNECTION + LAYER NORM  ("Add & Norm" in the paper's diagram)
# =============================================================================
#
#   output = LayerNorm(x + Sublayer(x))
#
# Residual (x + ...) lets gradients flow directly through the "+" during
# backprop no matter how deep the stack is (this is what makes 6+ stacked
# layers trainable at all - without it, gradients vanish/explode through
# repeated attention+FFN transformations). LayerNorm re-centers/rescales
# each token's feature vector to mean 0, variance 1 (then applies a learned
# scale/shift), which keeps activation magnitudes stable layer after layer.
class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_out):
        return self.norm(x + self.dropout(sublayer_out))


# =============================================================================
# 6. ENCODER LAYER  = self-attention + FFN, each wrapped in Add&Norm
# =============================================================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.addnorm2 = AddNorm(d_model, dropout)

    def forward(self, x, src_mask):
        attn_out, _ = self.self_attn(x, x, x, src_mask)   # every token looks at every other source token
        x = self.addnorm1(x, attn_out)
        ffn_out = self.ffn(x)
        x = self.addnorm2(x, ffn_out)
        return x


# =============================================================================
# 7. DECODER LAYER = masked self-attn + cross-attn + FFN, each Add&Norm'd
# =============================================================================
# The decoder has TWO attention sub-layers, not one:
#   (a) masked self-attention over the target sequence so far — masked so
#       position t can only see positions <= t (can't peek at the future
#       token it's trying to predict). This is the causal mask.
#   (b) cross-attention (aka "encoder-decoder attention") — queries come
#       from the decoder, but keys/values come from the ENCODER's output.
#       This is literally how information flows from source -> target:
#       each target position asks "which source tokens are relevant to
#       generate me?"
class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h)
        self.cross_attn = MultiHeadAttention(d_model, h)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.addnorm2 = AddNorm(d_model, dropout)
        self.addnorm3 = AddNorm(d_model, dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        self_attn_out, _ = self.self_attn(x, x, x, tgt_mask)          # (a) causal self-attention
        x = self.addnorm1(x, self_attn_out)
        cross_attn_out, _ = self.cross_attn(x, enc_out, enc_out, src_mask)  # (b) attend to encoder
        x = self.addnorm2(x, cross_attn_out)
        ffn_out = self.ffn(x)
        x = self.addnorm3(x, ffn_out)
        return x


# =============================================================================
# 8. FULL ENCODER / DECODER STACKS
# =============================================================================
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, h, d_ff, dropout=0.1, max_len=5000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, h, d_ff, dropout) for _ in range(N)])
        self.d_model = d_model

    def forward(self, src_ids, src_mask):
        # Embedding is scaled by sqrt(d_model) per the paper (section 3.4):
        # keeps embedding magnitude comparable to the positional encoding's.
        x = self.embed(src_ids) * math.sqrt(self.d_model)
        x = self.dropout(self.pos_enc(x))
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, h, d_ff, dropout=0.1, max_len=5000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, h, d_ff, dropout) for _ in range(N)])
        self.d_model = d_model

    def forward(self, tgt_ids, enc_out, src_mask, tgt_mask):
        x = self.embed(tgt_ids) * math.sqrt(self.d_model)
        x = self.dropout(self.pos_enc(x))
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x


# =============================================================================
# 9. FULL TRANSFORMER (encoder-decoder, as in the original paper)
# =============================================================================
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, N=6, h=8, d_ff=2048,
                 dropout=0.1, max_len=5000, pad_idx=0):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, h, d_ff, dropout, max_len)
        self.decoder = Decoder(tgt_vocab, d_model, N, h, d_ff, dropout, max_len)
        self.generator = nn.Linear(d_model, tgt_vocab)   # final Linear + softmax -> next-token probs
        self.pad_idx = pad_idx

    @staticmethod
    def make_padding_mask(ids, pad_idx):
        # (B, T) -> (B, 1, 1, T): broadcastable mask, 1 = attend, 0 = block
        return (ids != pad_idx).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def make_causal_mask(T, device):
        # lower-triangular matrix: position i may attend to positions <= i
        return torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)  # (1,1,T,T)

    def forward(self, src_ids, tgt_ids):
        device = src_ids.device
        src_mask = self.make_padding_mask(src_ids, self.pad_idx)                 # (B,1,1,T_src)
        tgt_pad_mask = self.make_padding_mask(tgt_ids, self.pad_idx)              # (B,1,1,T_tgt)
        causal_mask = self.make_causal_mask(tgt_ids.size(1), device)             # (1,1,T_tgt,T_tgt)
        tgt_mask = tgt_pad_mask & causal_mask.bool()                             # combine both

        enc_out = self.encoder(src_ids, src_mask)
        dec_out = self.decoder(tgt_ids, enc_out, src_mask, tgt_mask)
        logits = self.generator(dec_out)                                        # (B, T_tgt, vocab)
        return logits


# =============================================================================
# 10. SANITY-CHECK TRAINING RUN — toy "copy the reversed sequence" task
# =============================================================================
# This isn't translation, but it exercises every piece above end to end:
# embeddings, positional encoding, masked/unmasked attention, cross-attention,
# FFN, and teacher-forced training with a causal mask.
if __name__ == '__main__':
    torch.manual_seed(0)
    VOCAB = 12          # tokens 2..11 are "digits", 0 = <pad>, 1 = <bos>
    PAD, BOS = 0, 1

    def make_batch(batch_size, seq_len):
        src = torch.randint(2, VOCAB, (batch_size, seq_len))
        tgt_out = src.flip(dims=[1])                       # task: reverse the sequence
        tgt_in = torch.cat([torch.full((batch_size, 1), BOS), tgt_out[:, :-1]], dim=1)
        return src, tgt_in, tgt_out

    model = Transformer(src_vocab=VOCAB, tgt_vocab=VOCAB, d_model=64, N=2, h=4, d_ff=128,
                         dropout=0.1, pad_idx=PAD)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    model.train()
    for step in range(300):
        src, tgt_in, tgt_out = make_batch(batch_size=32, seq_len=8)
        logits = model(src, tgt_in)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), tgt_out.reshape(-1), ignore_index=PAD)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step:3d} | loss {loss.item():.4f}")

    # quick qualitative check
    model.eval()
    src, tgt_in, tgt_out = make_batch(batch_size=1, seq_len=8)
    with torch.no_grad():
        logits = model(src, tgt_in)
        pred = logits.argmax(-1)
    print("\nsource:            ", src.tolist()[0])
    print("target (reversed): ", tgt_out.tolist()[0])
    print("model prediction:  ", pred.tolist()[0])
