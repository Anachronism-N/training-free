import torch

from lifecycle_kv.role_episodic import (
    compute_head_role_evidence,
    masked_prompt_descriptor,
    select_dual_evidence_episode,
)


def test_masked_prompt_descriptor_ignores_padding():
    embeds = torch.tensor([[[3.0, 0.0], [1.0, 0.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])

    descriptor = masked_prompt_descriptor(embeds, mask)

    torch.testing.assert_close(descriptor, torch.tensor([[1.0, 0.0]]))


def test_dual_evidence_has_no_candidate_on_first_transition():
    decision = select_dual_evidence_episode(
        current_prompt_descriptor=torch.tensor([0.0, 1.0]),
        frame_prompt_descriptors=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        episode_ids=torch.tensor([0, 0]),
        visual_similarity=torch.tensor([[[0.2, 0.3], [0.1, 0.2]]]),
        current_episode_id=1,
        previous_episode_id=0,
    )

    assert not decision.accepted
    assert decision.abstain_reason == "no_nonrecent_episode"


def test_dual_evidence_selects_nonrecent_return_episode():
    decision = select_dual_evidence_episode(
        current_prompt_descriptor=torch.tensor([1.0, 0.0]),
        frame_prompt_descriptors=torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
        ),
        episode_ids=torch.tensor([0, 0, 1, 1]),
        visual_similarity=torch.tensor(
            [[[0.9, 0.8, 0.1, 0.0], [0.7, 0.8, 0.2, 0.1]]]
        ),
        current_episode_id=2,
        previous_episode_id=1,
    )

    assert decision.accepted
    assert decision.winner_episode_id == 0
    assert decision.cue_agreement


def test_dual_evidence_abstains_when_cues_disagree():
    decision = select_dual_evidence_episode(
        current_prompt_descriptor=torch.tensor([1.0, 0.0]),
        frame_prompt_descriptors=torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.7, 0.7], [0.7, 0.7], [0.0, 1.0]]
        ),
        episode_ids=torch.tensor([0, 0, 1, 1, 2]),
        visual_similarity=torch.tensor(
            [[[0.0, 0.1, 0.9, 0.8, 0.0], [0.1, 0.0, 0.8, 0.9, 0.0]]]
        ),
        current_episode_id=3,
        previous_episode_id=2,
    )

    assert not decision.accepted
    assert decision.abstain_reason == "cue_disagreement"


def test_head_role_evidence_prefers_persistent_head():
    q = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    memory_k = torch.tensor(
        [
            [[[1.0, 0.0], [1.0, 0.0]]],
            [[[1.0, 0.0], [1.0, 0.0]]],
            [[[1.0, 0.0], [1.0, 0.0]]],
        ]
    )
    memory_v = torch.tensor(
        [
            [[[1.0, 0.0], [1.0, 0.0]]],
            [[[1.0, 0.0], [-1.0, 0.0]]],
            [[[1.0, 0.0], [1.0, 0.0]]],
        ]
    )
    query_ema = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])

    evidence = compute_head_role_evidence(
        q, memory_k, memory_v, query_ema=query_ema
    )

    assert evidence.gate.shape == (1, 2)
    assert evidence.gate[0, 0] > evidence.gate[0, 1]
    assert evidence.role_codes[0, 0] == 0
    assert evidence.role_codes[0, 1] == 1


def test_head_role_single_frame_is_finite():
    q = torch.randn(1, 2, 3, 4)
    memory_k = torch.randn(1, 5, 3, 4)
    memory_v = torch.randn(1, 5, 3, 4)

    evidence = compute_head_role_evidence(q, memory_k, memory_v)

    assert torch.isfinite(evidence.gate).all()
    assert ((0.0 <= evidence.gate) & (evidence.gate <= 1.0)).all()


def test_relative_role_calibration_selects_top_evidence_head():
    q = torch.tensor([[[[1.0, 0.0]] * 4]])
    memory_k = torch.tensor(
        [
            [[[1.0, 0.0]] * 4],
            [[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]],
            [[[1.0, 0.0]] * 4],
        ]
    )
    memory_v = torch.tensor(
        [
            [[[1.0, 0.0]] * 4],
            [[[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]],
            [[[1.0, 0.0]] * 4],
        ]
    )
    query_ema = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]]]
    )

    evidence = compute_head_role_evidence(
        q,
        memory_k,
        memory_v,
        query_ema=query_ema,
        calibration="relative",
        keep_fraction=0.25,
        min_evidence_spread=0.01,
    )

    assert evidence.calibration_valid.item()
    assert evidence.gate[0, 0] > 0.5
    assert int((evidence.gate >= 0.5).sum().item()) == 1
    assert evidence.persistent_evidence[0, 0] > evidence.persistent_evidence[0, 1]


def test_role_calibration_fails_closed_without_evidence_spread():
    q = torch.tensor([[[[1.0, 0.0]] * 3]])
    memory_k = torch.tensor([[[[1.0, 0.0]] * 3]] * 3)
    memory_v = memory_k.clone()
    query_ema = torch.tensor([[[1.0, 0.0]] * 3])

    evidence = compute_head_role_evidence(
        q,
        memory_k,
        memory_v,
        query_ema=query_ema,
        calibration="hybrid",
        min_evidence_spread=0.01,
    )

    assert not evidence.calibration_valid.item()
    assert torch.count_nonzero(evidence.gate) == 0
