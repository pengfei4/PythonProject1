class Transformer(nn.Moude):
    '''整体模型'''
    def __init__(self,args):
        super().__init__()
        #必须输入词表大小和block size
        assert args.vocab_size is not None
        assert args.block_size is not None
        self.args = args
        self.transformer = nn.MoudleDict(dict(
            wte = nn.Embedding(args.vocab_size,args.n_embd),
            wpe = PositionalEncoding(args),
            drop = nn.Dropout(args.dropout),
            encoder = Encoder(args),
            decoder = Decoder(args),
        ))
        #最后的线性层，输入是n_embd,输出是词表大小
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        #初始化所有的权重
        self.apply(self._init_weights)
        #查看所有参数的数量
        print("number of parameters: %.2fm" % (self.get_num_params()/le6,))
    '''统计所有参数的数量'''
        def get_num_params(self, non_embedding = False):
            #non_embedding:是否统计embedding的参数
            n_params = sum(p.numel()for p in self.parameters())
            #如果不统计embedding的参数，就减去
            if non_embedding:
                n_params -= self.transformer.wte.weight.numel()
            return n_params
    '''初始化权重'''
        def _init_weights(self,module):
            #线性层和Embedding层初始化为正则分布
            if isinstance(module,nn.Linear):
                torch.nn.init.normal_(moudle.weight,mean=0.0,std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)

        def forward(self, idx ,target=None):
            #输入为idx,维度为(batch size,sequence length, 1);target为目标序列，用于计算loss
            device = idx.device
            b,t = idx.size()
            assert t <= self.args.block_size,f"不能计算该序列，该序列长度为{t},最大序列长度只有{self.args.block_size}"