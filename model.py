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
        norm_src = self.norm1(src) 
        
        if padding_mask is not None:
            padding_mask = padding_mask.bool()
            if src.size(1) != padding_mask.size(1):
                padding_mask = padding_mask[:, :src.size(1)]

            attn_output, _ = self.causalm_attn(
                query=norm_src, key=norm_src, value=norm_src, 
                key_padding_mask=padding_mask, attn_mask=causal_mask, is_causal=False
            )
        else:
            attn_output, _ = self.causalm_attn(
                query=norm_src, key=norm_src, value=norm_src, 
                attn_mask=causal_mask, is_causal=False
            )
        
        x = src + self.dropout_attn(attn_output)
        
        norm_ffn = self.norm2(x)
        ffn_out = self.ffn(norm_ffn)
        output = x + self.dropout_ffn(ffn_out)
        return output

class Decoder_SDPA(nn.Module):
    def __init__(self,d_model=1024,dim_feedforward=None,nheads=16,dropout=0.1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = int(8/3*d_model)
        self.dim_head = d_model//nheads
        self.nheads = nheads
        self.d_model = d_model

        self.norm1,self.norm2 = nn.RMSNorm(d_model), nn.RMSNorm(d_model)
        self.dropout_attn,self.dropout_ffn = nn.Dropout(dropout), nn.Dropout(dropout)

        self.q_proj = nn.Linear(d_model,d_model,bias=False)
        self.k_proj = nn.Linear(d_model,d_model,bias=False)
        self.v_proj = nn.Linear(d_model,d_model,bias=False)
        self.out_proj = nn.Linear(d_model,d_model,bias=False)
    
        self.ffn = SwiGLU(d_model,dim_feedforward)

    def forward(self,src):
        batch_size,seq_len,_ = src.shape
        src_norm = self.norm1(src)
        # Must convert bsz,seq_len,nheads,dim_head into bsz,dim_head,nheads,dim_head
        q_out = self.q_proj(src_norm).view(batch_size,seq_len,self.nheads,self.dim_head).transpose(1,2)
        k_out = self.k_proj(src_norm).view(batch_size,seq_len,self.nheads,self.dim_head).transpose(1,2)
        v_out = self.v_proj(src_norm).view(batch_size,seq_len,self.nheads,self.dim_head).transpose(1,2)

        attn_out = F.scaled_dot_product_attention(query=q_out,key=k_out,value=v_out,attn_mask=None,is_causal=True)
        attn_out = attn_out.transpose(1,2).reshape(batch_size,seq_len,self.d_model)

        attn_out = src+self.dropout_attn(self.out_proj(attn_out))
        attn_norm = self.norm2(attn_out)
        ffn_out = self.ffn(attn_norm)
        return attn_out+self.dropout_ffn(ffn_out)

class Transformer(nn.Module):
    def __init__(self,d_model=768,dim_feedforward=2048,nheads=16,n_layers=16,vocab_size=49_267,dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size,d_model)
        self.pos_embed = PosEncoder(embed_dim=d_model,max_length=5120)

        self.layers = nn.ModuleList([
            # Decoder(d_model,dim_feedforward,nheads,dropout)
            Decoder_SDPA(d_model,dim_feedforward,nheads,dropout)
            for _ in range(n_layers)
        ])

        self.out = nn.Linear(d_model,vocab_size)
        self.d_model = d_model
        self.final_norm = nn.RMSNorm(d_model)

    def forward(self,src,padding_mask=None):
        out = self.pos_embed(self.embed(src)*math.sqrt(self.d_model))

        dec_out = out
        for layer in self.layers:
            # dec_out = layer(dec_out,padding_mask) # When using the MHA implementation
            dec_out = layer(dec_out) # SDPA implementation doesn't take the padding_mask argument
        dec_out = self.final_norm(dec_out)
        logits = self.out(dec_out)
        return logits