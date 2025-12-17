import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiheadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super(MultiheadAttention,self).__init__()

        assert d_model % num_heads==0, "d_model must be divisible by num_heads(h)"


        #initilization of the parameter
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads #512/8 if h=8 

        self.W_q=nn.Linear(d_model,d_model) # for query
        self.W_k=nn.Linear(d_model,d_model) # for key
        self.W_v=nn.Linear(d_model,d_model)# for value
        self.W_o=nn.Linear(d_model,d_model) # for output

    def scaled_dot_product(self,Q,V,K,mask=None):
        #from fig:scaled_dot_prod
        # [....,seq_len,d_k]-->[....,d_k,seq_len]
        attn_scores=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.d_k)
        # from fig:fig1
        if mask is not None:
            attn_scores=attn_scores.masked_fill(mask==0,-1e9)

        attn_probs=torch.softmax(attn_scores,dim=-1)
        output=torch.matmul(attn_probs,V)

        return output
    
    def split_heads(self,x):
        batch_size,seq_length,d_model=x.size() 
        return x.view(batch_size,seq_length,self.num_heads,self.d_k).transpose(1,2) # returns ->(batch_size,self.num_heads,seq_len,self.d_k)
    
    def combine_heads(self,x):
        batch_size,_,seq_length,d_k=x.size()# (batch_size,self.num_heads,seq_len,self.d_k)
        return x.transpose(1,2).contiguous().view(batch_size,seq_length,self.d_model) #returns -> (batch_size,seq_len,self.d_model)
    
    def forward(self,Q,K,V,mask=None):
        Q=self.split_heads(self.W_q(Q))
        K=self.split_heads(self.W_k(K))
        v=self.split_heads(self.W_v(V))

        attn_output=self.scaled_dot_product(Q,K,V,mask)

        output=self.W_o(self.combine_heads(attn_output))
        return output
    


class PositionWiseFF(nn.Module):
    def __init__(self,d_model,d_ff): # d_ff=2048
        super(PositionWiseFF,self).__init__()
        self.fc1=nn.Linear(d_model,d_ff)
        self.fc2=nn.Linear(d_ff,d_model)
        self.relu=nn.Relu()

    def forward(self,x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_seq_length):
        super(PositionalEncoding,self).__init__()

        pe=torch.zeros(max_seq_length,d_model)
        position=torch.arange(0,max_seq_length,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*-(math.log(10000.0)/d_model))
    
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)

        self.register_buffer('pe',pe.unsqueeze(0))

    def forward(self,x):
        return x+self.pe[:,:x.size(1)]
    
class EncoderLayer(nn.Module):
    def __init(self,d_model,num_heads,d_ff,dropout):
        super(EncoderLayer,self).__init__()

        self.self_attn=MultiheadAttention(d_model,num_heads)
        self.ff=PositionWiseFF(d_model,d_ff)
        self.nomr1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,mask):
        attn_output=self.self_attn(x,x,x,mask)
        x=self.norm1(x+self.droput(attn_output))
        ff_output=self.ff(x)
        x=self.norm2(x+self.droput(ff_output))
        return x
    

class Decoder(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,droput):
        super(DecoderLayer,self).__init__()

        self.self_attn=MultiheadAttention(d_model,)