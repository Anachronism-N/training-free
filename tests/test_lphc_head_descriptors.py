"""Run these CPU-Torch integration tests on the server before GPU generation."""
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F
from lifecycle_kv.lphc import LPHCArchive, LPHCConfig, LPHCController, apply_lphc_attention
from test_lphc import _attention_inputs, _context, _ready_controller


def test_old_pooled_descriptor_is_bit_exact():
    v = torch.arange(4*3*2*5, dtype=torch.float32).reshape(4, 3, 2, 5)
    expected = F.normalize(torch.cat((v.mean((1, 2)), v.std((1, 2), unbiased=False)), -1), dim=-1, eps=1e-6)
    assert torch.equal(LPHCArchive._descriptors(v), expected)
    assert LPHCArchive._descriptors(v, "headwise").shape == (4, 2, 10)


def test_new_mode_requires_explicit_protocol():
    with pytest.raises(ValueError, match="protocol=v215"):
        LPHCConfig(alpha=.02, descriptor_mode="headwise")


@pytest.mark.parametrize("mode", ("pooled", "headwise", "headwise_centered"))
def test_random_is_descriptor_independent_and_rng_isolated(mode):
    old = _ready_controller(LPHCConfig(alpha=.02, mode="random", control_seed=9))
    new = _ready_controller(LPHCConfig(alpha=.02, mode="random", control_seed=9, protocol="v215", descriptor_mode=mode))
    context = _context(block=2, local=(6, 7))
    before = torch.random.get_rng_state().clone()
    a, b = old.retrieve(context), new.retrieve(context)
    assert torch.equal(a.frame_ids, b.frame_ids)
    assert torch.equal(a.k, b.k) and torch.equal(a.v, b.v)
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize("mode", ("headwise", "headwise_centered"))
def test_correct_scores_and_selection_frozen_within_block(mode):
    controller = _ready_controller(LPHCConfig(alpha=.02, protocol="v215", descriptor_mode=mode))
    history = controller.retrieve(_context(block=2, local=(6, 7)))
    stats = controller.last_selection_stats
    assert set(history.frame_ids.tolist()) == set(stats["ranked_frame_ids"][:4])
    assert stats["actual_choice"] and stats["candidate_count"] == 6
    assert all(torch.isfinite(torch.tensor(stats["ranked_scores"])))
    controller.previous_clean_descriptor.neg_()
    again = controller.retrieve(_context(block=2, phase=1, local=(6, 7)))
    assert torch.equal(again.frame_ids, history.frame_ids)


@pytest.mark.parametrize("mode", ("headwise", "headwise_centered"))
def test_zero_preserves_native_output_and_never_retrieves(mode):
    config = LPHCConfig(alpha=0., protocol="v215", descriptor_mode=mode)
    controller = LPHCController(config, layer_idx=0)
    y, q, k, v, _ = _attention_inputs()
    before = torch.random.get_rng_state().clone()
    result = apply_lphc_attention(y, q, k, v, context=_context(local=(0,)), config=config,
                                controller=controller, history_provider=lambda _: pytest.fail("retrieved at alpha0"),
                                attention_fn=lambda *_: pytest.fail("extra attention at alpha0"))
    assert result is y and torch.equal(before, torch.random.get_rng_state())
