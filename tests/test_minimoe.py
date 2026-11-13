import torch
import pytest

from mobilemoe.config import MiniMoEConfig
from mobilemoe.model import MiniMoEForCausalLM
from mobilemoe.tokenizer import ByteTokenizer


def test_model_logits_and_router_loss():
    config = MiniMoEConfig(context_length=32, d_model=32, n_layers=2, n_heads=4,
                           n_experts=4, experts_per_token=2, d_ff=64)
    model = MiniMoEForCausalLM(config)
    tokens = torch.randint(0, 256, (2, 12))
    logits, aux = model(tokens)
    assert logits.shape == (2, 12, 256)
    assert aux.ndim == 0
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux)


def test_forward_backward_has_gradients():
    config = MiniMoEConfig(context_length=16, d_model=32, n_layers=1, n_heads=4,
                           n_experts=4, experts_per_token=2, d_ff=64)
    model = MiniMoEForCausalLM(config)
    tokens = torch.randint(0, 256, (2, 8))
    logits, aux = model(tokens)
    loss = logits.mean() + aux * 0.01
    loss.backward()
    assert model.blocks[0].moe.router.weight.grad is not None


def test_context_limit():
    model = MiniMoEForCausalLM(MiniMoEConfig(context_length=8, d_model=32, n_heads=4,
                                             n_layers=1, d_ff=64))
    with pytest.raises(ValueError):
        model(torch.zeros((1, 9), dtype=torch.long))


def test_byte_tokenizer_roundtrip():
    tokenizer = ByteTokenizer()
    text = "MiniMoE سلام"
    assert tokenizer.decode(tokenizer.encode(text)) == text



def test_one_step_train_and_generate(tmp_path):
    from mobilemoe.generate import generate_text
    from mobilemoe.train import train

    text_path = tmp_path / "tiny.txt"
    text_path.write_bytes(("mini moe local training test " * 20).encode("utf-8"))
    checkpoint = tmp_path / "mini.pt"
    config = MiniMoEConfig(context_length=8, d_model=16, n_layers=1, n_heads=2,
                           n_experts=2, experts_per_token=1, d_ff=32)
    result = train(text_path, checkpoint, steps=1, batch_size=2, config=config)
    assert checkpoint.is_file()
    assert result["parameter_count"] > 0
    generated = generate_text(str(checkpoint), "Hi", max_new_tokens=2, temperature=0)
    assert generated.startswith("Hi")
