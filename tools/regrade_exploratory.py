"""Recompute the ten exploratory verdicts from public actions, without API calls."""
from pathlib import Path
import importlib.util
import json
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from regrade_public import canonical, file_hash, verify_visible_tools, outcomes, block_network
from benchmark_integrity import __main__ as replay
from benchmark_integrity.artifacts import source_hash


def main():
    sys.addaudithook(block_network)
    results = []
    for label in ("explicit-pressure", "grading-pressure"):
        folder = ROOT / "evidence" / "exploratory" / label
        index = json.loads((folder / "INDEX.json").read_text(encoding="utf-8"))
        plan = json.loads((folder / "plan.json").read_text(encoding="utf-8"))
        assert source_hash() == index["source_sha256"] == plan["source_sha256"]
        assert file_hash(folder / "recorded_adapter.py") == index["adapter_sha256"] == plan["extra_source_sha256"]
        spec = importlib.util.spec_from_file_location("recorded_" + label.replace("-", "_"), folder / "recorded_adapter.py")
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        records = []
        for entry, case in zip(index["episodes"], plan["cases"], strict=True):
            path = folder / entry["file"]
            assert file_hash(path) == entry["public_sha256"]
            artifact = json.loads(path.read_text(encoding="utf-8"))
            assert artifact["projection"]["raw_artifact_sha256"] == entry["raw_sha256"]
            assert artifact["source_sha256"] == index["source_sha256"]
            assert artifact["extra_source_sha256"] == index["adapter_sha256"]
            assert canonical(plan) == artifact["plan_sha256"]
            env = adapter.ContextEnvironment(case["seed"], case["condition"], case["target_accuracy"])
            assert canonical(env.prompt) == case["prompt_sha256"]
            assert canonical(env.tool_schemas()) == case["tools_sha256"]
            assert canonical(env.export()["dataset"]) == case["dataset_sha256"]
            assert canonical(env.reference_solution()) == canonical(case["reference_solution"])
            with patch.object(replay, "Environment", adapter.ContextEnvironment):
                grade = replay.verify_replay(artifact)
            tools = verify_visible_tools(artifact, adapter.ContextEnvironment)
            mechanical = outcomes(artifact["environment"], grade, artifact["status"], 1e-6)
            records.append({"seed": case["seed"], "status": artifact["status"], "outcomes": mechanical,
                            "reported_score": grade["reported_score"], "trusted_score": grade["trusted_score"],
                            "tool_evidence": tools})
        assert len(records) == 5
        results.append({"phase": index["phase"], "episodes": records})
    print(json.dumps({"status": "passed", "network": "disabled", "scope": "Public environment and visible tool replay; does not establish provider authorship or reproduce private full-history verification.", "phases": results}, indent=2))


if __name__ == "__main__":
    main()
