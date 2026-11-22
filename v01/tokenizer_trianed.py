import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path("/data/datasets/dataset")
TOKENIZER_JSON_PATH = BASE_DIR / "tokenizer.json"
TOKENIZER_DIR = BASE_DIR / "tokenizer"
CHAT_TEMPLATE_PATH = BASE_DIR / "chatml.jinja"
# (文件名, 文件总记录数, 最大抽样记录数)。均匀抽样可避免只取文件开头造成偏差。
DATA_SOURCES = [
    (DATASET_DIR / "pretrain_t2t_mini.jsonl", 1_270_238, 300_000),
    (DATASET_DIR / "sft_t2t.jsonl", 5_109_432, 50_000),
    (DATASET_DIR / "dpo.jsonl", 17_166, 10_000),
    (DATASET_DIR / "rlaif.jsonl", 19_502, 5_000),
    (DATASET_DIR / "agent_rl.jsonl", 39_988, 5_000),
    (DATASET_DIR / "agent_rl_math.jsonl", 20_000, 5_000),
]


def format_messages(messages: list[dict[str, Any]]) -> str:
    """按 `chatml.jinja` 的规则把对话记录转换为训练文本。"""
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        reasoning = message.get("reasoning_content", "")
        tools = message.get("tools", "")

        parts.append(f"<|im_start|>{role}\n")
        if tools:
            parts.append(f"{tools}\n")
        if reasoning:
            parts.append(f"<think>\n{reasoning}\n</think>\n")
        parts.append(f"{content}<|im_end|>\n")
    return "".join(parts)


def record_to_texts(record: dict[str, Any]) -> Iterator[str]:
    """从预训练、SFT、DPO 和 RL 记录中提取可训练文本。"""
    text = record.get("text")
    if isinstance(text, str) and text.strip():
        yield text

    conversations = record.get("conversations")
    if isinstance(conversations, list):
        rendered = format_messages(conversations)
        if rendered:
            yield rendered

    for key in ("chosen", "rejected"):
        messages = record.get(key)
        if isinstance(messages, list):
            rendered = format_messages(messages)
            if rendered:
                yield rendered

    # Agent 数据中的 `gt` 是奖励校验用答案，不是模型实际接收的对话序列。
    # 它通常已经出现在 conversations 的最后一条 assistant 消息中，单独加入会重复采样，
    # 并且会丢失 ChatML 角色边界，因此 tokenizer 训练只消费 conversations。


def train_iterator(
    sources: list[tuple[Path, int, int]],
) -> Iterator[str]:
    """逐行均匀抽样 JSONL，限制 BPE 统计规模并避免额外内存占用。"""
    for file_path, total_records, max_records in sources:
        if not file_path.is_file():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        if total_records <= 0 or max_records <= 0:
            continue

        sample_count = min(total_records, max_records)
        # 用整数区间判断是否命中样本，单次扫描即可均匀覆盖整个文件。
        previous_bucket = -1
        selected_records = 0
        with file_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if line_number > total_records or selected_records >= sample_count:
                    break
                bucket = (line_number - 1) * sample_count // total_records
                if bucket == previous_bucket:
                    continue
                previous_bucket = bucket
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"JSON 解析失败: {file_path}:{line_number}: {error.msg}"
                    ) from error

                if not isinstance(record, dict):
                    raise ValueError(
                        f"记录必须是 JSON 对象: {file_path}:{line_number}"
                    )
                yield from record_to_texts(record)
                selected_records += 1

        print(f"已抽样 {file_path.name}: {selected_records}/{total_records} 条记录")
#tokenizer训练的步骤
'''
1、规划流水线
    1.1、选择算法
    1.2、清洗数据nfkc---->归一化
    1.3、预切分---->将数据切分成最小单元
    1.4、解码--->再将切分后的数据转换成字符串
2、加载数据
    2.1、加载数据
    2.2、切分数据
    2.3、保存切分后的数据
3、训练
    3.1、加载切分后的数据
    3.2、训练
    3.3、保存模型
4、测试
    4.1、加载模型
    4.2、测试
'''
#流水线
tok=Tokenizer(models.BPE())#选择算法，字节合并算法，在训练类似gpt模型的时候使用这个算法，因为gpt模型是字节级别的，所以使用字节合并算法
#tok = Tokenizer(models.BPE(unk_token="<unk>"))
tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)#预切分，将数据切分成最小单元
'''
ByteLevel--->将数据切分成最小单元，目的：先给文本划好边界，BPE 只在每个片段内部合并，不能跨片段合并。
如果不切，语料里 dog.、dog!、dog? 这种高频组合会被 BPE 合成一个 token，
词表里会塞满"单词+标点"的组合，浪费空间，而且泛化能力差。切开之后，dog 和 . 各自独立，组合起来就能覆盖所有情况。
预切分还能大幅加快 BPE 训练（只需要统计片段内部的相邻字符对）。
① 用正则切片段（默认是 GPT-2 的正则）：字母连成一段、数字连成一段、标点单独一段，空格归到后面那个词的开头。
② 把每个片段转成 UTF-8 字节，再把每个字节映射成一个看得见的字符
add_prefix_space:是否在切分后的数据前面添加空格，默认不用，因为数据量够大，所以不用添加空格
'''
tok.decoder=decoders.ByteLevel()#解码，再将切分后的数据转换成字符串
'''
ByteLevel:将切分后的数据转换成字符串
'''
#tok.normalizer=normalizers.NFKC()#归一化，将数据转换成标准格式,因为这次的数据是整理之后的，因此不需要做归一化
'''
NFKC:将数据转换成标准格式,把写法不同、含义相同的字符统一成一种。

输入	               NFKC 之后
ＡＢＣ１２３（全角）	ABC123
①	                  1
ﬁ（连字）	           fi

不做归一化，１ 和 1 会被当成两个不同的东西，各占词表位置。
但它也会丢信息（比如 ① 变成 1 后就还原不回去了），所以很多模型（GPT-2、Llama 3）干脆不做归一化，
Qwen 用的是 NFC，比 NFKC 温和。
'''

# 控制类 token 会被 `skip_special_tokens=True` 删除，适合消息边界、模态占位符
# 和保留槽位。列表顺序决定它们训练后的固定 ID，前三项必须保持不变。
special_tokens = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<|audio_pad|>",
    "<tts_pad>",
    "<tts_text_bos>",
    "<tts_text_eod>",
    "<tts_text_bos_single>",
] + [f"<|reserved_{i}|>" for i in range(16)]

# 解析类标签也必须各占一个 token，但不能设为 special，否则开启
# `skip_special_tokens` 后标签会在 reasoning/tool parser 处理前被删除。
parser_tokens = [
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
]

trainer = trainers.BpeTrainer(
    vocab_size=6400,
    special_tokens=special_tokens,
    min_frequency=5,
    show_progress=True,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # 收录全部 256 个字节
)
tok.train_from_iterator(
    iterator=train_iterator(DATA_SOURCES),
    trainer=trainer,
)  # 流式解析 JSONL 后训练

# 训练结束后再注册普通 AddedToken：它们会获得连续的新 ID、保持整体编码，且不会被
# `skip_special_tokens=True` 删除。模型词嵌入大小必须使用最终 `len(hf)`。
_ = tok.add_tokens(parser_tokens)
tok.save(str(TOKENIZER_JSON_PATH))  # 先保存底层 tokenizer，避免后续封装失败导致训练结果丢失

# BPE 训练完成后 `tok` 只负责字符串与 token ID 的转换；下面封装为 Hugging Face
# tokenizer，补充标准 token 职责、最大长度和 chat template，供训练代码及 vLLM 使用。
# 当前 `special_tokens` 的顺序使前三个 token 的 ID 固定为 0、1、2：
# - <|endoftext|>：PAD/UNK，ID 0；
# - <|im_start|>：消息起始边界并登记为 BOS，ID 1；
# - <|im_end|>：消息结束边界并登记为 EOS，ID 2。
# vLLM 默认读取 `eos_token_id`，因此模型生成 <|im_end|> 后即可停止。
# `model_max_length` 必须与模型配置实际支持的上下文长度保持一致。
hf = PreTrainedTokenizerFast(
    tokenizer_object=tok,
    bos_token="<|im_start|>",
    eos_token="<|im_end|>",
    pad_token="<|endoftext|>",
    unk_token="<|endoftext|>",
    model_max_length=32768,
)

# `chat_template` 已显式写入每条消息的边界，因此不再自动添加 BOS/EOS，避免重复。
# 完整 SFT/DPO 会话使用 `add_generation_prompt=False`；推理或 RL prompt 使用
# `add_generation_prompt=True`，只追加 assistant 起始边界，等待模型生成正文和 EOS。
hf.add_bos_token = False
hf.add_eos_token = False

# `parser_tokens` 已在底层 tokenizer 中以普通 AddedToken 注册，HF 封装会保留其
# ID 和普通属性，因此这里不能再次改变其 special 状态。
# 模板会写入保存目录的 `chat_template.jinja`；训练阶段应通过
# `apply_chat_template(..., add_generation_prompt=False)` 生成完整会话，推理阶段设为 True。
hf.chat_template = CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")
hf.save_pretrained(TOKENIZER_DIR)







