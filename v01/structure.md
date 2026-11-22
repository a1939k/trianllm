### 文件结构
zerotransformer.py   模型网络和 forward
train.py             训练循环
generate.py          文本生成循环

### 实际调用流程
messages
   ↓ tokenizer.apply_chat_template
带 ChatML 标记的 prompt
   ↓ tokenizer 编码
input_ids
   ↓ generate 循环
output_ids
   ↓ tokenizer.decode
最终文本

例如当前 tokenizer：
用户消息
    ↓
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
    ↓
转换为 token IDs
    ↓
ZeroTransformerModel.generate()
    ↓
生成到 <|im_end|>（ID 2）时停止
    ↓
decode


如果你以后用 vLLM：

客户端请求
   ↓
vLLM 自己执行生成循环
   ↓
调用模型 forward
   ↓
采样、KV Cache、EOS 检测、批处理调度


### dataset改如何做

第一阶段（现在）
├── PretrainDataset
├── DataLoader
├── 训练循环
└── 验证 loss 能否下降

第二阶段
├── SFTDataset
├── assistant-only labels
└── SFT 训练循环

第三阶段
├── DPODataset
└── DPO loss

第四阶段
├── RL/Agent Dataset
├── rollout
├── reward
└── 强化学习更新
