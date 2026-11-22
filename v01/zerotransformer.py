import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

import math
@dataclass
class ZeroTransformerConfig:
    batch_size: int = 16
    learning_rate: float = 3e-4
    max_sequence_length: int = 2048
    num_layers: int = 12
    num_heads: int = 8
    hidden_size: int = 1024
    embeed_size:int = 1024
    hidden_size_per_head:int=hidden_size//num_heads
    dropout: float = 0.1
#    activation_fn: str = "relu"
#    tie_embeddings: bool = True
#    tie_projection: bool = True
    vocab_size: int = 6400
'''
模型读取一个 token 和预测这个 token 时使用同一套词向量，通常有助于训练稳定性和泛化。
对于小模型来说这个可以减少一份参数，从而减少模型大小。
大致规律： 小模型通常打开（如 Qwen2.5-0.5B/1.5B、Llama-3.2-1B/3B、Gemma）。大模型通常关闭（如 Llama-2/3-7B+、Qwen2.5-7B+），因为此时词表矩阵在总参数里占比小，分开反而让输入和输出各自更灵活。
'''
#下面开始写模型结构
#先来实现一个单头
class ZeroTransformerSingleHead(nn.Module):
    def __init__(self, config: ZeroTransformerConfig):
        super().__init__()
        self.query=nn.Linear(config.hidden_size,config.hidden_size_per_head)
        self.key=nn.Linear(config.hidden_size,config.hidden_size_per_head)
        self.value=nn.Linear(config.hidden_size,config.hidden_size_per_head)
        self.dropout=nn.Dropout(config.dropout)
        self.hidden_size_per_head=config.hidden_size_per_head
        self.hidden_size=config.hidden_size
        #使用register_buffer来加速后面的运算，他的作用就是生成一个下三角矩阵，然后将掩码注册到buffer区中，这个buffer中的参数不会被更新，随着to(device)移动，默认保存到state_dict中
        self.register_buffer("attention_mask",torch.tril(
            torch.ones(config.max_sequence_length,
                            config.max_sequence_length)))
    def forward(self,x):
        #x:batch_size,sequence_length,hidden_size
        #q:batch_size,sequence_length,hidden_size_per_head
        #k:batch_size,sequence_length,hidden_size_per_head
        #v:batch_size,sequence_length,hidden_size_per_head
        batch_size,sequence_length,hidden_size=x.size()
        q=self.query(x)
        k=self.key(x)
        v=self.value(x)
        weight = q @ k.transpose(-2, -1)   # @ 就是 torch.matmul 的简化写法
        #因为是自回归模型，这个是需要进行mask处理的,qk之后的weight形状是[batch_size,sequence_length,sequence_length]
        weight=weight.masked_fill(self.attention_mask[:sequence_length,:sequence_length]==0,float('-inf'))
        weight=weight/math.sqrt(self.hidden_size_per_head)
        weight=F.softmax(weight,dim=-1)
        weight=self.dropout(weight)
        out=weight @ v
        return out

class ZeroTransformerMultiHead(nn.Module):
    def __init__(self, config: ZeroTransformerConfig):
        super().__init__()
        self.heads=nn.ModuleList([ZeroTransformerSingleHead(config) for _ in range(config.num_heads)])
        #输入的就是多头拼接后的结果，然后进行一个线性映射，映射到hidden_size，也就是拼接后还要再经过一个线性变换再当成最后的数据传递给下一层
        self.projection=nn.Linear(config.hidden_size,config.hidden_size)
        self.dropout=nn.Dropout(config.dropout)
    def forward(self,x):
        out=torch.cat([head(x) for head in self.heads],dim=-1)
        out=self.projection(out)
        out=self.dropout(out)
        return out

class ZeroTransformerFNN(nn.Module):
    def __init__(self, config: ZeroTransformerConfig):
        super().__init__()
        self.feed_forward_net=nn.Sequential(
            nn.Linear(config.hidden_size,config.hidden_size*4),
            nn.ReLU(),
            nn.Linear(config.hidden_size*4,config.hidden_size),
            nn.Dropout(config.dropout)
        )
    def forward(self,x):
        out=self.feed_forward_net(x)
        return out

#上面就是对一个block的拆解，下面把他们组合起来
class ZeroTransformerBlock(nn.Module):
    def __init__(self, config: ZeroTransformerConfig):
        super().__init__()
        self.atten=ZeroTransformerMultiHead(config)
        self.feed_forward=ZeroTransformerFNN(config)
        self.fn1=nn.LayerNorm(config.hidden_size)
        self.fn2=nn.LayerNorm(config.hidden_size)
    def forward(self, x):
        #进入到mha的时候需要进行一个layernorm，然后进行一个残差链接就是 residual connection
        x=x+self.atten(self.fn1(x))
        #从mha出来后要过一个fnn之前，要做一次layernorm,然后进行一个残差链接 residual connection
        x=x+self.feed_forward(self.fn2(x))
        return x


#完整的gpt模型
class ZeroTransformerModel(nn.Module):
    def __init__(self, config: ZeroTransformerConfig):
        super().__init__()
        self.token_embeeding_talbe=nn.Embedding(config.vocab_size,config.embeed_size)
        self.position_embeeding_table=nn.Embedding(config.max_sequence_length,config.embeed_size)
        self.blocks=nn.ModuleList([ZeroTransformerBlock(config) for _ in range(config.num_layers)])
        #在所有 Transformer Block 之后，用于把最终的隐藏向量变成“下一个 token 的预测分数”
        #token ID → Embedding → 多个 Transformer Block → self.fn → self.lm_head → logits
        self.fn=nn.LayerNorm(config.hidden_size)
        #最后的线性层，将hidden_size映射到vocab_size
        self.lm_head=nn.Linear(config.hidden_size,config.vocab_size)
        #初始化权重，遍历模型自身以及所有子模块，对每个模块调用一次 self._init_weights，统一初始化参数。他的传输参数是module
        #假设模型包含 Embedding、多个 Block、lm_head，apply 会进入每个 Block，继续遍历里面的 Attention、FFN、Linear 等模块。
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module,nn.Linear):
            #正态分布初始化
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
    def forward(self, x,target=None):
        batch_size,sequence_length=x.size()
        token_embeed=self.token_embeeding_talbe(x)
        position_embeed=self.position_embeeding_table(
            torch.arange(sequence_length,device=x.device)
        )
        x=token_embeed+position_embeed
        #流转到各个block中
        for block in self.blocks:
            x=block(x)
        x=self.fn(x)
        #下面就是预测的下一个token
        logits=self.lm_head(x)#[batch_size,sequence_length,vocab_size]
        #这个就是如果target是None，说明是预测，否则是训练，训练的时候需要计算loss
        if target is None:
            loss=None
        else:
            batch, seq_len, vocab_size = logits.size()
            #使用交叉熵损失来计算loss，那么就需要将logits和target进行reshape，然后再计算loss
            #交叉熵损失要求的形状是(预测,目标)，预测形状是[样本数, 类别数]，目标形状是[样本数]，所以需要将logits和target进行reshape   
            # #样本数 = batch × seq_len类别数 = vocab_size
            # 为什么使用view而不是reshape？因为view是inplace操作，不会产生新的tensor，而reshape是out-of-place操作，会产生新的tensor
            # 而且view的速度比reshape快，因为view是直接修改了tensor的shape，而reshape是重新分配了内存空间，然后再进行赋值。
            # 所以使用view是更高效的，因为view不需要重新分配内存空间，只需要修改shape即可。view不改变里面的数据和排列顺序
            logits=logits.view(batch*seq_len,vocab_size)
            target=target.view(batch*seq_len)
            loss=F.cross_entropy(logits,target)
        return logits,loss




