from pathlib import Path

from transformers import AutoTokenizer


TOKENIZER_DIR = Path(__file__).with_name("tokenizer")


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    expected_tokens = {
        "bos_token": ("<|im_start|>", 1),
        "eos_token": ("<|im_end|>", 2),
        "pad_token": ("<|endoftext|>", 0),
        "unk_token": ("<|endoftext|>", 0),
    }
    for attribute, (expected_token, expected_id) in expected_tokens.items():
        actual_token = getattr(tokenizer, attribute)
        actual_id = getattr(tokenizer, f"{attribute}_id")
        assert actual_token == expected_token, (
            f"{attribute} 错误：期望 {expected_token!r}，实际 {actual_token!r}"
        )
        assert actual_id == expected_id, (
            f"{attribute}_id 错误：期望 {expected_id}，实际 {actual_id}"
        )

    assert tokenizer.model_max_length == 32_768, (
        f"model_max_length 错误：实际为 {tokenizer.model_max_length}"
    )

    parser_tokens = [
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
    ]
    for token in parser_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        assert tokenizer.encode(token, add_special_tokens=False) == [token_id], (
            f"{token!r} 没有被编码为单个 token"
        )
        assert token not in tokenizer.all_special_tokens, (
            f"{token!r} 不应登记为 HF 特殊 token，否则解码时可能被删除"
        )
        assert tokenizer.decode([token_id], skip_special_tokens=True) == token, (
            f"{token!r} 在 skip_special_tokens=True 时应保留供解析器识别"
        )

    messages = [
        {"role": "system", "content": "你是一个助手。"},
        {"role": "user", "content": "你好"},
        {
            "role": "assistant",
            "reasoning_content": "用户正在打招呼。",
            "content": "你好，有什么可以帮助你？",
        },
    ]
    expected_training_text = (
        "<|im_start|>system\n你是一个助手。<|im_end|>\n"
        "<|im_start|>user\n你好<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n用户正在打招呼。\n</think>\n"
        "你好，有什么可以帮助你？<|im_end|>\n"
    )
    training_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    assert training_text == expected_training_text, (
        "训练模板渲染结果不一致：\n"
        f"期望：{expected_training_text!r}\n"
        f"实际：{training_text!r}"
    )

    tool_messages = [
        {
            "role": "system",
            "content": "你可以使用工具。",
            "tools": '[{"name":"calculate"}]',
        },
        {"role": "user", "content": "计算 1+1"},
    ]
    expected_tool_text = (
        "<|im_start|>system\n"
        '[{"name":"calculate"}]\n'
        "你可以使用工具。<|im_end|>\n"
        "<|im_start|>user\n计算 1+1<|im_end|>\n"
    )
    tool_text = tokenizer.apply_chat_template(
        tool_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    assert tool_text == expected_tool_text, (
        "工具定义模板渲染结果不一致：\n"
        f"期望：{expected_tool_text!r}\n"
        f"实际：{tool_text!r}"
    )

    prompt_messages = messages[:-1]
    expected_generation_prompt = (
        "<|im_start|>system\n你是一个助手。<|im_end|>\n"
        "<|im_start|>user\n你好<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    generation_prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    assert generation_prompt == expected_generation_prompt, (
        "推理模板渲染结果不一致：\n"
        f"期望：{expected_generation_prompt!r}\n"
        f"实际：{generation_prompt!r}"
    )
    assert not generation_prompt.endswith(tokenizer.eos_token), (
        "推理 prompt 不应提前以 EOS 结束"
    )

    training_encoding = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    training_ids = training_encoding["input_ids"]
    assert training_ids[-2] == tokenizer.eos_token_id, (
        "完整训练会话末尾换行之前应包含 <|im_end|>"
    )
    assert tokenizer.decode(training_ids, skip_special_tokens=False) == training_text, (
        "tokenize 后再 decode 的结果与模板文本不一致"
    )

    print("Tokenizer 配置与 ChatML 模板测试全部通过。")
    print(f"BOS: {tokenizer.bos_token!r} -> {tokenizer.bos_token_id}")
    print(f"EOS: {tokenizer.eos_token!r} -> {tokenizer.eos_token_id}")
    print(f"PAD: {tokenizer.pad_token!r} -> {tokenizer.pad_token_id}")
    print(f"最大上下文长度: {tokenizer.model_max_length}")
    print("推理 prompt：")
    print(generation_prompt)


if __name__ == "__main__":
    main()
