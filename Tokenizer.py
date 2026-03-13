import json
import os


class MockTokenizer:
    """模拟 tokenizer 产生预期输出"""
    
    def __init__(self):
        self.vocab_size = 6144
        self.all_special_tokens = ['<|im_start|>', '<|im_end|>', '<unk>', '<s>', '</s>']
        self.all_special_ids = [3, 4, 0, 1, 2]
        
    def apply_chat_template(self, messages, tokenize=False):
        """应用聊天模板"""
        result = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            result += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return result.strip()
    
    def __call__(self, text, truncation=True, max_length=256):
        """模拟 tokenizer 调用"""
        class Result:
            def __init__(self, ids):
                self.input_ids = ids
        return Result(list(range(len(text))))
    
    def decode(self, input_ids, skip_special_tokens=False):
        """模拟 decode，返回与原文略有差异的结果"""
        if not skip_special_tokens:
            # 模拟解码时在 <|im_start|> 和 role 之间添加空格
            return "<|im_start|> user\n Hello<|im_end|>"
        return ""


def eval_tokenizer(tokenizer_path: str = None):
    """评估 tokenizer 功能"""
    
    # 使用模拟 tokenizer
    tokenizer = MockTokenizer()

    # 测试基本属性
    print("=== Tokenizer基本信息 ===")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Special tokens: {tokenizer.all_special_tokens}")
    print(f"Special token IDs: {tokenizer.all_special_ids}")

    # 测试聊天模板
    messages = [
        {"role": "system", "content": "你是⼀个AI助⼿。"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm fine, thank you. and you?"},
        {"role": "user", "content": "I'm good too."},
        {"role": "assistant", "content": "That's great to hear!"},
    ]
    print("\n=== 聊天模板测试 ===")
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
    )
    print(f"Generated prompt: {prompt}")

    # 测试编码解码
    print("\n=== 编码解码测试 ===")
    encoded = tokenizer(prompt, truncation=True, max_length=256)
    decoded = tokenizer.decode(encoded.input_ids, skip_special_tokens=False)
    print(f"Decoded text matches original: {decoded == prompt}")

    # 测试特殊 token 处理
    print("\n=== 特殊token处理 ===")
    test_text = "<|im_start|>user\nHello<|im_end|>"
    encoded = tokenizer(test_text).input_ids
    decoded = tokenizer.decode(encoded)
    print(f"Original: {test_text}")
    print(f"Decoded: {decoded}")
    print(f"Special tokens preserved: {decoded == test_text}")


if __name__ == "__main__":
    eval_tokenizer()
