import torch
import torch.nn as nn

from src.modeling.common.rope import apply_rotary_pos_emb, ROPE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def test_rope_forward_shape():
    rope = ROPE(dim=16, base=10000).to(device=DEVICE)
    seq_len = 10
    sin, cos = rope(seq_len=seq_len, device=DEVICE, has_cls_token=False)
    assert sin.shape == (seq_len, 16)
    assert cos.shape == (seq_len, 16)

def test_rope_forward_with_cls_token():
    rope = ROPE(dim=16, base=10000).to(device=DEVICE)
    seq_len = 10
    sin, cos = rope(seq_len=seq_len, device=DEVICE, has_cls_token=True)
    assert sin.shape == (seq_len, 16)
    assert cos.shape == (seq_len, 16)
    assert torch.all(sin[0] == 0)
    assert torch.all(cos[0] == 1)

def test_apply_rotary_pos_emb_shape():
    rope = ROPE(dim=16, base=10000).to(device=DEVICE)
    seq_len = 10
    sin, cos = rope(seq_len=seq_len, device=DEVICE, has_cls_token=False)

    q = torch.randn(2, 4, seq_len, 16, device=DEVICE)
    k = torch.randn(2, 4, seq_len, 16, device=DEVICE)

    q_rot, k_rot = apply_rotary_pos_emb(q=q, k=k, sin=sin, cos=cos, rotary_dim=16)

    assert q_rot.shape == (2, 4, seq_len, 16)
    assert k_rot.shape == (2, 4, seq_len, 16)

def test_apply_rotary_pos_emb_values():
    rope = ROPE(dim=16, base=10000).to(device=DEVICE)
    seq_len = 10
    sin, cos = rope(seq_len=seq_len, device=DEVICE, has_cls_token=False)

    q = torch.randn(2, 4, seq_len, 16, device=DEVICE)
    k = torch.randn(2, 4, seq_len, 16, device=DEVICE)

    q_rot, k_rot = apply_rotary_pos_emb(q=q, k=k, sin=sin, cos=cos, rotary_dim=16)

    assert not torch.allclose(q_rot, q)
    assert not torch.allclose(k_rot, k)