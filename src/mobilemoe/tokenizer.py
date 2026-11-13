"""A deliberately simple byte-level tokenizer (no external tokenizer files)."""
class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(int(token) & 0xFF for token in tokens).decode("utf-8", errors="replace")
