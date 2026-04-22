import torch
import torch.nn as nn
import math 


"""
Implementation notes !
implements LoRA for a linear layer
formula = h = Wx + (alpha/rank)*(B @ A @ x) 
in-implementation : h = Wx + (x @ A.T @ B.T) * (alpha/rank)

Where
W: pre-trained frozen weights. Most of the time, we freeze early layers of network and tune the last few layers of the network
A: low rank matrix (rank, in_dim), initailized with kaiming uniform, 
B: low rank matrix (out_dim, rank), initailized with zeros
x : is the input tensor
"""

class LoraLayer(nn.Module):
    """implement the lora layer"""
    def __init__(
        self, 
        in_dim, 
        out_dim, 
        rank=4, 
        alpha=32
    ): # here meta-parameters values are set as-per the Table 11
        super().__init__()
        
        #A-matrix initialized with empty and it shape (rank, in_dim)
        self.A=nn.Parameter(torch.empty(rank, in_dim))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        #B-matrix initialized with zeros and its shape (out_dim, rank)
        self.B=nn.Parameter(torch.zeros(out_dim, rank))

        #scaling factor: alpha /rank
        self.scaling = alpha/rank

    def forward(self, x):
        #result : (x @ A.T @ B.T) * scaling
        return (x @ self.A.t() @ self.B.t()) * self.scaling


class GPT2LoraWrapper(nn.Module):
    """replace the weight (Wq) query and value (Wv) projection in GPT-2"""
    def __init__(
        self, 
        original_layer, 
        rank=4, 
        alpha=32
    ):
        super().__init__()
        self.original_layer = original_layer
        
        #freeze the original weights W
        self.original_layer.weight.requires_grad = False

        #embedding_dim for GPT-2 Conv1D is accessed via .nx
        self.embedding_dim=original_layer.nx

        #define LoRA modules for query (q) and value (v)
        self.lora_q=LoraLayer(self.embedding_dim, self.embedding_dim, rank, alpha)
        self.lora_v=LoraLayer(self.embedding_dim, self.embedding_dim, rank, alpha)
        
    def forward(self, x):
        #get the original output (batch, seq, 3*embedding_dim)
        qkv=self.original_layer(x)

        #seperate the slices 
        q, k, v = torch.split(qkv, self.embedding_dim, dim=-1)

        #now, apply LoRA updates 
        q_updated = q + self.lora_q(x)
        v_updated = v + self.lora_v(x)

        # concat back together: q, k(original), v
        return torch.cat([q_updated, k, v_updated], dim=-1)

#here is the next taks - which is inject LoRA to the GPT-2
def inject_lora(model, rank=4, alpha=32):
    """here walk through the attention blocks and replace the cross_attention (c_attn) with lora wrapper"""
    #here this below loop is must, because it keeps all the params freeze initially
    for param in model.parameters():
        param.requires_grad=False
        
    for i in range(len(model.transformer.h)):
        target=model.transformer.h[i].attn.c_attn
        model.transformer.h[i].attn.c_attn = GPT2LoraWrapper(target, rank, alpha)
        
    return model