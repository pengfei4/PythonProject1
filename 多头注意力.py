import torch.nn as nn
import torch
'''多头注意力计算模块'''
class MultiHeadAttention(nn.Module):
    def __init__(self,args:ModelArgs,is_causal=False):
        #构造函数
        #args：配置对象
        super().__init__()
        #隐藏层维度必须是头数的整数倍，因为后面我们会将输入拆成头数个矩阵
        assert args.dim%args.n_heads == 0
        #每个头的维度，等于模型维度除以头的总数。
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads

        #Wq,Wk,Wv参数矩阵，每个参数矩阵为n_embd x dim
        #这里铜鼓三个组合来代替了n个参数矩阵的组合，其逻辑在于矩阵内积再拼接其实等于拼接矩阵再内积，
        #每一个线性层其实相当于n个参数矩阵的拼接
        self.wq = nn.Linear(args.n_embd,self.n_head *self.head_dim,bias=False)
        self.wk = nn.Linear(args.n_embd,self.n_head *self.head_dim,bias=False)
        self.wv = nn.Linear(args.n_embd,self.n_head *self.head_dim,bias=False)
        #输出权重矩阵，维度为dim x dim (head_dim = dim / n_heads)
        self.wo = nn.Linear(self.n_heads * self.head_dim,args.dim,bias=False)
        #注意力的dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        #残差连接的dropout
        self.resid_dropout = nn.Dropout(args.dropout)
        self.is_causal - is_causal

        #创建一个上三角矩阵，用于遮蔽未来信息
        #注意，多头注意力，Mask矩阵比之前我们定义的多一个维度
        if is_causal:
            mask = torch.full((1,1,args.max_seq_len,args.max_seq_len),float("-inf"))
            mask = torch.triu(mask,diagonal=1)
            #注册为模型缓冲区
            self.register_buffer("mask",mask)

    def forward(self,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor):
        #获取批次大小和序列长度，[batch_size,seq-len,dim]
        bsz,seqlen, _ = q.shape
        #计算查询Q，键K，,值V，输入通过参数矩阵层，维度为(B,T,n_embed) x (n_embed,dim) -> (B,T,dim)
        xq,xk,xv = self.wq(q),self.wk(k),self.wv(v)
        #将Q，K，V差分成多头，维度为（B,T,n_head,dim // n_head）,然后交换维度，变成(B,n_head, T, dim // n_head)
        #因为在注意力计算中我们是取了后两个维度参与计算
        #为什么要先按B*T*n_head*C//n_head展开再互换1、2维度而不是直接按注意力输入展开，是因为view的展开方式是直接把输入全部排开
        #然后按要求构造，可以发现只有上述操作能够实现我们将每个头对应部分取出来的目标
        xq = xq.view(bsz,seqlen,self.n_local_heads,self.head_dim)
        xk = xk.view(bsz,seqlen,self.n_local_heads,self.head_dim)
        xv = xv.view(bsz,seqlen,self.n_local_heads,self.head_dim)
        xq = xq.transpose(1,2)
        xk = xk.transpose(1,2)
        xv = xv.transpose(1,2)

        #注意力计算
        #计算QK^T / sqrt(d_k),维度为（B,nh,T,hs） x (B,nh,hs,T) -> (B,nh,T,T)
        scores = torch.matmul(xq,xk.transpose(2,3)) / math.sqrt(self.head_dim)
        #掩码自注意力必须有注意力掩码
        if self.is_causal:
            assert hasattr(self,'mask')
            #这里截取到序列长度，因为有些序列可能比max_seq_len短
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
        #计算softmax，维度为（B,nh,T,T）
        scores = F.softmax(scores.float(),dim=-1).type_as(xq)
        #做Dropout
        scores = self.attn_dropout(scores)
        #V * Score,维度为(B,nh,T,T) x (B,nh,T,hs) -> (B,nh,T,hs)
        output = torch.matmul(scores,xv)

        #恢复时间维度并合并头
        #将多头的结果拼接起来，先交换维度为(B,T,n_head,C // n_head),再拼接成(B,T,n_head * C // n_head)
        #contigious函数用于重新开辟一块新内存存储，因为pytorch设置先transpose再view会报错
        #因为view直接基于底层存储得到，然而transpose并不会改变底层存储，因此需要额外内存
        output = output.transpose(1,2).contiguous().view(bsz,seqlen,-1)

        #最终投影回残差流
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output



class MLP(nn.Module):
    '''前反馈神经网络'''
    def __init__(self,dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        #定义第一层线性变换，从输入维度到隐藏维度
        self.w1 = nn.linear(dim, hidden_dim, bias=False)
        #定义第第二层线性变换，从隐藏维度到输入维度
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        #定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        #前向传播函数
        #首先，输入x通过第一层线性变换和RELU激活函数
        #最后，通过第二层线性变换和dropout层
        return self.dropout(self.w2(F.relu(self.w1(x))))
