import json
from pathlib import Path
from typing import Any, BinaryIO, cast

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class PretrainDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """把 JSONL 纯文本转换成用于 next-token prediction 的训练样本。"""

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_sequence_length: int,
        max_samples: int | None = None,
        add_bos_token: bool = False,
    ) -> None:
        # 第 1 步：保存数据路径，并确认预训练 JSONL 文件确实存在。
        self.data_path = Path(data_path).expanduser().resolve()
        if not self.data_path.is_file():
            raise FileNotFoundError(f"找不到预训练数据文件：{self.data_path}")

        # 第 2 步：保存已经训练好的 tokenizer；Dataset 只使用它，不会重新训练词表。
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.add_bos_token = add_bos_token
        if self.max_sequence_length < 1:
            raise ValueError("max_sequence_length 必须大于 0")

        # 第 3 步：取得特殊 token ID，避免在数据代码中硬编码 0、1、2。
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id
        bos_token_id = tokenizer.bos_token_id
        if not isinstance(eos_token_id, int):
            raise ValueError("tokenizer 必须配置整数 eos_token_id")
        if not isinstance(pad_token_id, int):
            raise ValueError("tokenizer 必须配置整数 pad_token_id")
        if self.add_bos_token and not isinstance(bos_token_id, int):
            raise ValueError("add_bos_token=True 时 tokenizer 必须配置整数 bos_token_id")
        self.eos_token_id: int = eos_token_id
        self.pad_token_id: int = pad_token_id
        self.bos_token_id: int | None = bos_token_id if isinstance(bos_token_id, int) else None

        # 第 4 步：只记录每一条 JSONL 的字节位置，不把 1GB 以上文本全部载入内存。
        # 有了 offset，DataLoader 即使 shuffle，也可以按索引随机读取任意一行。
        self.line_offsets = self._build_line_offsets(max_samples)

        # 第 5 步：文件句柄延迟到 __getitem__ 再打开；每个 DataLoader worker 各自持有一个。
        self._file: BinaryIO | None = None

    def _build_line_offsets(self, max_samples: int | None) -> list[int]:
        """扫描 JSONL，记录每个非空行的起始字节位置。"""
        if max_samples is not None and max_samples < 1:
            raise ValueError("max_samples 必须大于 0 或设为 None")

        offsets: list[int] = []
        with self.data_path.open("rb") as file:
            while max_samples is None or len(offsets) < max_samples:
                # tell() 是本行开始的位置，之后可以通过 seek() 精确返回这里。
                offset = file.tell()
                line = file.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)

        if not offsets:
            raise ValueError(f"数据文件中没有可用记录：{self.data_path}")
        return offsets

    def _get_file(self) -> BinaryIO:
        """为当前进程或 DataLoader worker 创建自己的二进制文件句柄。"""
        if self._file is None or self._file.closed:
            self._file = self.data_path.open("rb")
        return self._file

    def __len__(self) -> int:
        # 第 6 步：DataLoader 通过它得知可以采样多少条文本。
        return len(self.line_offsets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 第 7 步：根据索引跳到对应 JSONL 行，只读取当前样本。
        file = self._get_file()
        file.seek(self.line_offsets[index])
        line = file.readline().decode("utf-8")

        # 第 8 步：解析 JSON，并取出预训练数据中的 text 字段。
        try:
            record = cast(dict[str, Any], json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"第 {index} 条记录不是合法 JSON") from error
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"第 {index} 条记录缺少非空字符串 text 字段")

        # 第 9 步：调用本地 tokenizer 完成“文本 -> token ID”；关闭自动特殊 token，
        # 由 Dataset 明确决定 BOS/EOS 的位置，避免 tokenizer 再添加一遍。
        prefix_length = 1 if self.add_bos_token else 0
        max_text_length = self.max_sequence_length - prefix_length
        token_ids = cast(
            list[int],
            self.tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_text_length,
            ),
        )

        # 第 10 步：可选地添加 BOS，并在每条文档末尾添加 EOS。
        # 总长度最多是 max_sequence_length + 1，因为构造 x/y 需要多取一个 token。
        full_ids: list[int] = []
        if self.add_bos_token:
            if self.bos_token_id is None:
                raise RuntimeError("BOS token 配置在初始化后意外丢失")
            full_ids.append(self.bos_token_id)
        full_ids.extend(token_ids)
        full_ids.append(self.eos_token_id)

        # 第 11 步：将同一序列左右错开一位，构造“根据当前 token 预测下一个 token”。
        # 例如 [10, 20, 30, EOS] 会得到 x=[10,20,30]、y=[20,30,EOS]。
        input_ids = full_ids[:-1]
        labels = full_ids[1:]

        # 第 12 步：input 使用 PAD 补到固定长度，让 DataLoader 能直接堆叠 batch；
        # labels 的补齐位置使用 -100，交叉熵会忽略这些位置，不让 PAD 产生 loss。
        padding_length = self.max_sequence_length - len(input_ids)
        input_ids.extend([self.pad_token_id] * padding_length)
        labels.extend([-100] * padding_length)

        # 第 13 步：Embedding 和交叉熵标签都需要 torch.long 类型的整数 Tensor。
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    def __getstate__(self) -> dict[str, object]:
        # 第 14 步：多进程 DataLoader 创建 worker 时不传递已打开的文件句柄。
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __del__(self) -> None:
        # 第 15 步：Dataset 销毁时关闭当前进程持有的文件句柄。
        if self._file is not None and not self._file.closed:
            self._file.close()
