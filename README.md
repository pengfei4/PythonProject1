# AI 大模型算法学习项目

> 本项目是学习 AI 大模型算法时的代码实现总结，涵盖了从基础 Transformer 架构到 Agent 智能体的完整知识体系。

## 📁 项目结构

```
PythonProject1/
├── 核心模型实现/
│   ├── LLMA2.py           # LLaMA2 模型完整实现
│   ├── t1.py              # 基础 Transformer 实现
│   ├── transform.py        # Transformer 架构
│   ├── 多头注意力.py       # 多头注意力机制
│   ├── attention.py        # Attention 详细实现与调试
│   ├── RMSNorm.py         # RMSNorm 层归一化
│   └── RunPractice.py      # 语言模型训练实践
│
├── 分词器/
│   ├── Tokenizer.py       # 分词器实现与测试
│   └── test_tokenizer/    # 分词器配置目录
│
├── 智能体系统/
│   ├── tools.py           # 工具执行器（SerpAPI 搜索）
│   ├── React.py           # ReAct 智能体实现
│   ├── agent.py           # OpenAI 兼容 LLM 客户端
│   ├── available_tools.py # 可用工具定义
│   └── ModelConfig.py     # 模型配置
│
├── 实战应用/
│   ├── search_attraction.py    # 景点搜索工具
│   ├── Weather_Free_Search.py  # 天气查询工具
│   └── OpenAICompatibleClient.py # OpenAI 兼容客户端
│
├── 学习示例/
│   └── day5/             # 第五天学习内容
│       ├── test1.py      # Encoder-Decoder Transformer
│       └── test2.py       # 语言模型实现
│
└── HTML 示例/
    ├── 百里屠苏简介.html  # HTML 基础练习
    ├── 自荐信.html        # 段落格式化
    ├── 全国计算机.html    # 列表综合应用
    └── sy2-1~5.html      # HTML 学习系列
```

## 🧠 核心知识点

### 1. Transformer 架构

#### 多头注意力 (Multi-Head Attention)
- 多个注意力头并行计算
- 允许模型同时关注不同位置的不同表示子空间
- 核心公式：$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$

#### 位置编码 (Positional Encoding)
- 引入序列位置信息
- 包含正弦/余弦位置编码和可学习位置编码

#### 前馈网络 (Feed Forward Network)
- 两层线性变换 + 激活函数 (GELU/SwiGLU)
- 隐藏层维度通常是输入的 4 倍

### 2. LLaMA2 模型特色

#### RMSNorm
- 只使用均方根进行归一化
- 比 LayerNorm 更轻量，效果相当
- 公式：$\frac{x}{RMS(x)} \cdot \gamma$

#### Rotary Position Embedding (RoPE)
- 旋转位置编码
- 通过旋转矩阵实现位置感知
- 优点：可扩展至长文本

#### SwiGLU 激活函数
- Swish + Gated Linear Unit
- 比 ReLU/GELU 效果更好

#### KV Cache
- 键值对缓存加速推理
- 自回归生成的关键优化

### 3. 分词器 (Tokenizer)

#### BPE (Byte-Pair Encoding)
- 字节对编码算法
- 平衡词表大小和未登录词问题

#### 特殊 Token
- `<|im_start|>` / `<|im_end|>` - 对话分隔
- `<s>` / `</s>` - 序列边界
- `<unk>` - 未知词

#### Chat Template
- 聊天模板格式化
- 多轮对话支持

### 4. ReAct 智能体

#### 核心思想
- **Reason**: LLM 推理当前状态
- **Act**: 选择并执行工具
- **Observation**: 观察工具返回结果
- 循环直到得到最终答案

#### 工具系统
- 搜索工具 (SerpAPI)
- 天气查询
- 景点搜索

## 🚀 快速开始

### 环境配置

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # Linux/Mac

# 安装依赖
pip install torch transformers tokenizers
```

### 运行示例

#### 1. 测试 LLM 客户端
```bash
python agent.py
```

#### 2. 运行 ReAct 智能体
```bash
python React.py
```

#### 3. 测试分词器
```bash
python Tokenizer.py
```

#### 4. 训练语言模型
```bash
python RunPractice.py
```

## 📦 依赖包

```
torch>=2.0.0
transformers
tokenizers
python-dotenv
google-search-results
numpy
```

## 📝 环境变量配置

在 `.env` 文件中配置：

```env
LLM_API_KEY=your-api-key
LLM_MODEL_ID=qwen-plus
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
SERPAPI_API_KEY=your-serpapi-key
```

## 🎯 学习路径建议

1. **入门阶段**
   - 理解 Transformer 架构原理
   - 掌握多头注意力机制
   - 学习位置编码

2. **进阶阶段**
   - 实现完整 Transformer 模型
   - 理解 LayerNorm/RMSNorm
   - 掌握 Rotary Position Embedding

3. **实战阶段**
   - 构建 LLM 客户端
   - 实现工具调用系统
   - 开发 ReAct 智能体

4. **应用阶段**
   - 微调预训练模型
   - 部署推理服务
   - 优化推理速度

## 📚 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [RoFormer: Enhanced Position Embedding](https://arxiv.org/abs/2104.09864)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

---

> 本项目仅供学习参考，代码实现可能存在不足之处，欢迎指正和改进。
