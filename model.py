import torch, math
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self,d_model,d_ffn):
        super().__init__()
        self.gate_v = nn.Linear(d_model,d_ffn*2,bias=False)
        self.out = nn.Linear(d_ffn,d_model,bias=False)

    def forward(self,x):
        gate, value = self.gate_v(x).chunk(2,dim=-1)
        return self.out(F.silu(gate)*value)

class PosEncoder(nn.Module):
    def __init__(self,embed_dim=1024,max_length=5120):
        super().__init__()

        pe = torch.zeros(max_length,embed_dim)
        position = torch.arange(0,max_length,dtype=torch.float).unsqueeze(1)

        div = torch.exp(torch.arange(0,embed_dim,2).float()*(-math.log(10000)/embed_dim))

        pe[:,0::2] = torch.sin(position*div) # Sin for even positions
        pe[:,1::2] = torch.cos(position*div) # Cos for odd positions

        self.register_buffer('pe',pe.unsqueeze(0))

    def forward(self,x):
        x = x+self.pe[:,:x.size(1),:]
        return x

class Decoder(nn.Module):
    def __init__(self,d_model=1024,dim_feedforward=None,nheads=16,dropout=0.1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = int((8/3*d_model))
        self.causalm_attn = nn.MultiheadAttention(d_model,nheads,dropout=dropout,batch_first=True)
        self.norm1,self.norm2 = nn.RMSNorm(d_model), nn.RMSNorm(d_model)
        self.dropout_attn,self.dropout_ffn = nn.Dropout(dropout), nn.Dropout(dropout)

        self.ffn = SwiGLU(d_model=d_model,d_ffn=dim_feedforward)

    def forward(self, src, padding_mask=None):
        seq_len = src.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=src.device) == 0, diagonal=1).bool()
        norm_src = self.norm1(src) # Or use src depending on your pre-norm/post-norm preference
        
        if padding_mask is not None:
            padding_mask = padding_mask.bool()
            if src.size(1) != padding_mask.size(1):
                padding_mask = padding_mask[:, :src.size(1)]

            attn_output, _ = self.causalm_attn(
                query=src, key=src, value=src, 
                key_padding_mask=padding_mask, attn_mask=causal_mask, is_causal=False
            )
        else:
            attn_output, _ = self.causalm_attn(
                query=src, key=src, value=src, 
                attn_mask=causal_mask, is_causal=False
            )
        
        # Now attn_output is guaranteed to be a Tensor
        x = src + self.dropout_attn(attn_output)
        
        norm_ffn = self.norm2(x)
        ffn_out = self.ffn(norm_ffn)
        output = x + self.dropout_ffn(ffn_out)
        return output

class Transformer(nn.Module):
    def __init__(self,d_model=1024,dim_feedforward=4096,nheads=16,n_layers=8,vocab_size=49_267,dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size,d_model)
        self.pos_embed = PosEncoder(embed_dim=d_model,max_length=5120)

        self.layers = nn.ModuleList([
            Decoder(d_model,dim_feedforward,nheads,dropout)
            for _ in range(n_layers)
        ])

        self.out = nn.Linear(d_model,vocab_size)
        self.d_model = d_model

    def forward(self,src,padding_mask=None):
        out = self.pos_embed(self.embed(src)*math.sqrt(self.d_model))

        dec_out = out
        for layer in self.layers:
            dec_out = layer(dec_out,padding_mask)
        
        logits = self.out(dec_out)
        return logits