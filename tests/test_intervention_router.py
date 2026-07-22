import json

import torch

from lifecycle_kv.intervention_router import (
    InterventionRouterState,
    InterventionRoutingConfig,
    OfflineInterventionProfile,
    route_memory_intervention,
)


def _inputs(heads: int = 4):
    q = torch.ones(1, 2, heads, 2)
    native = torch.ones(1, 2, heads, 2)
    memory = native.clone()
    memory[:, :, 2] *= -1
    memory[:, :, 3] *= 8
    confidence = torch.tensor([[0.9, 0.7, 0.5, 0.3]])
    margin = torch.tensor([[0.8, 0.6, 0.4, 0.2]])
    entropy = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    accepted = torch.ones(1, heads, dtype=torch.bool)
    return q, native, memory, confidence, margin, entropy, accepted


def test_online_router_selects_safe_high_utility_heads():
    q, native, memory, confidence, margin, entropy, accepted = _inputs()
    decision = route_memory_intervention(
        q=q,
        query_reference=None,
        native_output=native,
        memory_output=memory,
        confidence=confidence,
        retrieval_margin=margin,
        retrieval_entropy=entropy,
        accepted=accepted,
        base_gate=0.05,
        fusion_mode="residual",
        layer_idx=10,
        memory_mode="noisy",
        attention_call_index=0,
        config=InterventionRoutingConfig(
            mode="online",
            head_budget_fraction=0.5,
            max_delta_to_native=0.2,
            min_utility_spread=0.0,
        ),
        state=InterventionRouterState(),
    )

    assert decision.selected.tolist() == [[True, True, False, False]]
    assert decision.abstain_reason is None
    assert decision.alignment[0, 2] < 0
    assert decision.delta_to_native[0, 3] <= 0.05


def test_hybrid_router_uses_offline_profile(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "num_layers": 30,
                "num_heads": 4,
                "entries": [
                    {
                        "layer": 10,
                        "head": head,
                        "utility": utility,
                        "reliability": 1.0,
                    }
                    for head, utility in enumerate([0.1, 0.2, 0.9, 0.8])
                ],
            }
        ),
        encoding="utf-8",
    )
    profile = OfflineInterventionProfile.load(path)
    q, native, memory, confidence, margin, entropy, accepted = _inputs()
    memory[:, :, 2] = native[:, :, 2]
    memory[:, :, 3] = native[:, :, 3]
    decision = route_memory_intervention(
        q=q,
        query_reference=None,
        native_output=native,
        memory_output=memory,
        confidence=confidence,
        retrieval_margin=margin,
        retrieval_entropy=entropy,
        accepted=accepted,
        base_gate=0.01,
        fusion_mode="residual",
        layer_idx=10,
        memory_mode="noisy",
        attention_call_index=0,
        config=InterventionRoutingConfig(
            mode="hybrid",
            head_budget_fraction=0.5,
            min_utility_spread=0.0,
        ),
        state=InterventionRouterState(),
        offline_profile=profile,
    )

    assert decision.selected.tolist() == [[False, False, True, True]]
    assert decision.offline_utility is not None


def test_router_abstains_when_heads_are_indistinguishable():
    heads = 4
    q = torch.ones(1, 2, heads, 2)
    output = torch.ones_like(q)
    equal = torch.full((1, heads), 0.5)
    decision = route_memory_intervention(
        q=q,
        query_reference=None,
        native_output=output,
        memory_output=output,
        confidence=equal,
        retrieval_margin=equal,
        retrieval_entropy=equal,
        accepted=torch.ones(1, heads, dtype=torch.bool),
        base_gate=0.01,
        fusion_mode="residual",
        layer_idx=0,
        memory_mode="noisy",
        attention_call_index=0,
        config=InterventionRoutingConfig(
            mode="online",
            min_utility_spread=0.05,
        ),
        state=InterventionRouterState(),
    )

    assert not decision.selected.any()
    assert decision.abstain_reason == "insufficient_utility_spread"
