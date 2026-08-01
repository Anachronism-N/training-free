import json

from scripts.build_v152_online_policy_suite import write_suite


def test_v152_uses_only_prompts_unseen_by_v150_and_v151(tmp_path):
    prompts = tmp_path / "moviebench.txt"
    prompts.write_text(
        "\n".join(f"A diverse long-video prompt number {index}." for index in range(128))
        + "\n",
        encoding="utf-8",
    )
    v150 = tmp_path / "v150.json"
    v151 = tmp_path / "v151.json"
    v150.write_text(json.dumps({"source_prompt_indices": list(range(32))}))
    v151.write_text(json.dumps({"source_prompt_indices": list(range(32, 64))}))
    signed = tmp_path / "signed.json"
    signed.write_text(
        json.dumps(
            {
                "maps": {
                    "low4": {str(layer): [0, 1, 2, 3] for layer in range(30)},
                    "middle4": {
                        str(layer): [4, 5, 6, 7] for layer in range(30)
                    },
                    "high4": {
                        str(layer): [8, 9, 10, 11] for layer in range(30)
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "suite"
    metadata = write_suite(
        output,
        natural_prompts=prompts,
        v150_suite_metadata=v150,
        v151_suite_metadata=v151,
        signed_map_path=signed,
    )
    assert metadata["job_count"] == 128
    assert metadata["source_prompt_indices"] == list(range(64, 128))
    assert metadata["expected_downstream_records_per_profile"] == 100
    jobs = [
        json.loads(line)
        for line in (output / "v152_core_128.jsonl").read_text().splitlines()
    ]
    assert {job["source_prompt_index"] for job in jobs} == set(range(64, 128))
    plan = json.loads((output / "v152_probe_plan.json").read_text())
    assert len(plan["probes"]) == 24
    assert sum("head_selector" in probe for probe in plan["probes"]) == 12
    assert all(
        len(probe.get("head_map", {})) == 30
        for probe in plan["probes"]
        if "head_map" in probe
    )
