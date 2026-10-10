"""Random-weight tiny Qwen3 target + tiny DFlash drafter, for CPU tests through the real code paths."""
import torch
from dflash.model import DFlashDraftModel
from transformers import Qwen3Config, Qwen3ForCausalLM

VOCAB = 8  # small vocab so random drafts get partial acceptance
MASK_ID = VOCAB - 1


def tiny_target(seed=0, dtype=torch.float32, init_std=1.0):
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=6,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=4096,
                      tie_word_embeddings=True, initializer_range=init_std)
    cfg._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(cfg).to(dtype).eval()
    return model


def tiny_drafter(target, seed=1, dtype=torch.float32, init_std=1.0):
    torch.manual_seed(seed)
    tc = target.config
    cfg = Qwen3Config(vocab_size=VOCAB, hidden_size=tc.hidden_size, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=4096,
                      initializer_range=init_std, num_target_layers=tc.num_hidden_layers, block_size=16,
                      dflash_config={"mask_token_id": MASK_ID, "target_layer_ids": [1, 3]})
    cfg._attn_implementation = "sdpa"
    model = DFlashDraftModel(cfg).to(dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
