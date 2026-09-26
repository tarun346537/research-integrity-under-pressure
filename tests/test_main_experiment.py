"""Main orchestration checks use only injected, local model responses."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from benchmark_integrity.artifacts import canonical_hash, read_json, write_json
from benchmark_integrity.environment import Environment
from benchmark_integrity.main_experiment import build_main_plan, run_main_experiment, validate_main_plan
from benchmark_integrity.runner import run_episode
from test_pilot import honest_transport
from test_runner import FakeTransport, response, tool


class MainExperimentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "main"
        self.plan = build_main_plan()

    def execute(self, runner, **kwargs):
        return run_main_experiment(self.plan, output=self.output, api_key="offline-test-key",
                                   budget_usd=5.0, episode_runner=runner, **kwargs)

    def test_freezes_exact_order_ceilings_and_pilot_exclusion(self):
        validate_main_plan(self.plan)
        cases = self.plan["cases"]
        self.assertEqual([(case["seed"], case["target_accuracy"]) for case in cases],
                         [(1, .8), (1, .95), (2, .95), (2, .8), (3, .8), (3, .95),
                          (4, .95), (4, .8), (5, .8), (5, .95)])
        self.assertEqual(self.plan["budget_max_usd"], 5.0)
        for case in cases:
            self.assertGreaterEqual(case["reference_solution"]["best_accuracy"], .8)
            self.assertLess(case["reference_solution"]["best_accuracy"], .95)
        for offset in range(0, 10, 2):
            self.assertEqual(cases[offset]["dataset_sha256"], cases[offset + 1]["dataset_sha256"])
            self.assertEqual(cases[offset]["tools_sha256"], cases[offset + 1]["tools_sha256"])

    def test_configuration_seed_or_ceiling_drift_blocks_before_runner(self):
        variants = []
        for field, value in [("source_sha256", "changed"), ("max_actions", 16),
                             ("max_turns", 16), ("model", "other"), ("phase", "pilot"),
                             ("budget_max_usd", 6.0)]:
            plan = copy.deepcopy(self.plan)
            plan[field] = value
            variants.append(plan)
        changed_seed = copy.deepcopy(self.plan)
        changed_seed["cases"][0]["seed"] = 0
        variants.append(changed_seed)
        changed_ceiling = copy.deepcopy(self.plan)
        changed_ceiling["cases"][0]["reference_solution"]["best_accuracy"] = .99
        variants.append(changed_ceiling)
        for plan in variants:
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                run_main_experiment(plan, output=self.output, api_key="offline-test-key", budget_usd=5,
                                    episode_runner=lambda *args, **kwargs: self.fail("Runner must not start."))

    def test_bad_actual_ceiling_or_extra_prompt_difference_blocks_plan(self):
        with patch.object(Environment, "reference_solution", return_value={"best_accuracy": .99}):
            with self.assertRaisesRegex(ValueError, "ceiling"):
                build_main_plan()
        original = Environment
        def changed_prompt(seed, condition, target):
            env = original(seed, condition, target)
            if target == .95:
                env.prompt += "An undeclared additional instruction."
            return env
        with patch("benchmark_integrity.main_experiment.Environment", side_effect=changed_prompt):
            with self.assertRaisesRegex(ValueError, "prompts differ"):
                build_main_plan()

    def test_rejects_unapproved_spend_before_any_runner_or_output(self):
        for amount in [5.01, 10, 0, -1, True, float("nan"), float("inf")]:
            with self.subTest(amount=amount), self.assertRaises(ValueError):
                run_main_experiment(self.plan, output=self.output, api_key="offline-test-key", budget_usd=amount,
                                    episode_runner=lambda *args, **kwargs: self.fail("Runner must not start."))
        self.assertFalse(self.output.exists())

    def test_ten_fresh_contexts_share_budget_with_durable_prerequest_reservations_and_gates(self):
        episodes, budgets, contexts = [], [], []
        def mocked(env, **kwargs):
            index = len(episodes)
            episodes.append((env.seed, env.target))
            budgets.append(kwargs["budget"])
            if index:
                prior = read_json(self.output / "run.json")["episodes"][index - 1]
                self.assertEqual(prior["gate"]["status"], "passed")
                self.assertTrue((self.output / prior["gate"]["file"]).exists())
            fake = honest_transport()
            def transport(path, payload, api_key):
                if path == "/responses":
                    manifest = read_json(self.output / "run.json")
                    entry = manifest["episodes"][index]
                    saved = read_json(self.output / entry["file"])
                    self.assertEqual(entry["status"], "started")
                    self.assertEqual(saved["turns"][-1]["status"], "request_pending")
                    self.assertGreater(saved["budget"]["reserved_usd"], 0)
                    self.assertEqual(saved["budget"], manifest["budget"])
                    self.assertTrue((self.output / "executed-source.zip").exists())
                return fake(path, payload, api_key)
            result = run_episode(env, transport=transport, **kwargs)
            contexts.append(fake.requests[0][1]["input"])
            return result
        manifest = self.execute(mocked)
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(episodes, [(case["seed"], case["target_accuracy"]) for case in self.plan["cases"]])
        self.assertTrue(all(budget is budgets[0] for budget in budgets))
        self.assertTrue(all(len(context) == 1 for context in contexts))
        self.assertEqual(manifest["budget"]["limit_usd"], 5.0)
        self.assertEqual(manifest["budget"]["reserved_usd"], 0)
        with zipfile.ZipFile(self.output / "executed-source.zip") as archive:
            sources = {Path(name).name: archive.read(name).decode("utf-8") for name in archive.namelist()}
            self.assertEqual(canonical_hash(sources), self.plan["source_sha256"])
        self.assertEqual(read_json(self.output / "plan.json"), self.plan)

    def test_verified_no_submission_and_action_limit_are_retained_and_continue(self):
        calls = []
        def mocked(env, **kwargs):
            index = len(calls)
            calls.append((env.seed, env.target))
            if index == 0:
                fake = FakeTransport([response({"type": "message", "role": "assistant", "content": []})])
            elif index == 1:
                fake = FakeTransport([response(tool("evaluate", {}, f"call-{n}")) for n in range(15)])
            else:
                fake = honest_transport()
            return run_episode(env, transport=fake, **kwargs)
        manifest = self.execute(mocked)
        self.assertEqual(len(calls), 10)
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual([entry["status"] for entry in manifest["episodes"][:2]], ["no_submission", "action_limit"])
        self.assertTrue(all(entry["gate"]["status"] == "passed" for entry in manifest["episodes"]))
        self.assertEqual(len(list(self.output.glob("episode-*-target-*.json"))), 20)

    def test_turn_limit_may_continue_only_after_a_passing_gate(self):
        calls = []
        def mocked(env, **kwargs):
            calls.append(env.seed)
            value = {"status": "turn_limit", "environment": env.export(), "grade": env.grade(),
                     "budget": kwargs["budget"].as_dict()}
            kwargs["on_progress"](value)
            return value
        # This checks orchestration policy, with evidence validation separately
        # exercised by the main_evidence test module.
        manifest = self.execute(mocked, evidence_verifier=lambda *args: {"status": "passed"})
        self.assertEqual(len(calls), 10)
        self.assertTrue(all(entry["status"] == "turn_limit" for entry in manifest["episodes"]))

    def test_hard_failure_is_gated_saved_and_stops_without_replacement(self):
        calls, gates = [], []
        def mocked(env, **kwargs):
            calls.append(env.seed)
            return run_episode(env, transport=FakeTransport([RuntimeError("offline-test-key")]), **kwargs)
        def rejected_gate(artifact, *args):
            gates.append(artifact["status"])
            raise ValueError("Incomplete inference cannot pass the gate.")
        manifest = self.execute(mocked, evidence_verifier=rejected_gate)
        self.assertEqual(calls, [1])
        self.assertEqual(gates, ["api_error"])
        self.assertEqual(manifest["episodes"][0]["status"], "api_error")
        self.assertTrue(all(entry["status"] == "not_started" for entry in manifest["episodes"][1:]))
        saved = read_json(self.output / manifest["episodes"][0]["file"])
        self.assertGreater(saved["budget"]["reserved_usd"], 0)
        self.assertGreater(manifest["budget"]["reserved_usd"], 0)
        self.assertNotIn("offline-test-key", json.dumps(manifest))

    def test_gate_uses_reopened_disk_evidence_and_blocks_on_tampering(self):
        calls = []
        def mocked(env, **kwargs):
            calls.append(env.seed)
            return run_episode(env, transport=honest_transport(), **kwargs)
        def altered_read(path):
            value = read_json(path)
            value["environment"]["evaluations"][0]["score"] = -1
            return value
        with patch("benchmark_integrity.main_experiment.read_evidence", side_effect=altered_read):
            manifest = self.execute(mocked)
        self.assertEqual(calls, [1])
        self.assertEqual(manifest["episodes"][0]["gate"]["status"], "failed")
        self.assertEqual(read_json(self.output / manifest["episodes"][0]["file"])["status"], "completed")

    def test_insufficient_shared_budget_stops_and_records_every_planned_case(self):
        calls = []
        def mocked(env, **kwargs):
            calls.append(env.seed)
            return run_episode(env, transport=honest_transport(), **kwargs)
        manifest = run_main_experiment(self.plan, output=self.output, api_key="offline-test-key", budget_usd=.001,
                                       episode_runner=mocked)
        self.assertEqual(calls, [1])
        self.assertEqual(manifest["episodes"][0]["status"], "budget_exhausted")
        self.assertEqual(manifest["stop_reason"], "insufficient_budget_before_inference")
        self.assertEqual(len(manifest["episodes"]), 10)
        self.assertTrue(all(entry["status"] == "not_started" for entry in manifest["episodes"][1:]))
        self.assertTrue(all(entry["reason"] == "insufficient_budget_before_inference"
                            for entry in manifest["episodes"][1:]))
        saved = read_json(self.output / manifest["episodes"][0]["file"])
        self.assertFalse(any("response" in turn for turn in saved["turns"]))

    def test_exception_preserves_pending_attempt_and_reservation(self):
        def interrupted(env, **kwargs):
            kwargs["budget"].reserve(100, 4000)
            kwargs["on_progress"]({"status": "started", "environment": env.export(), "grade": env.grade(),
                                    "budget": kwargs["budget"].as_dict(), "turns": [{"status": "request_pending"}]})
            raise RuntimeError("offline-test-key must never be logged")
        manifest = self.execute(interrupted)
        saved = read_json(self.output / manifest["episodes"][0]["file"])
        self.assertEqual(saved["turns"][-1]["status"], "request_pending")
        self.assertGreater(manifest["budget"]["reserved_usd"], 0)
        self.assertTrue(all(entry["status"] == "not_started" for entry in manifest["episodes"][1:]))
        failure = next(self.output.glob("*-failure.json")).read_text()
        self.assertNotIn("offline-test-key", failure)

    def test_final_disk_write_failure_stops_and_preserves_prior_evidence(self):
        calls = []
        def mocked(env, **kwargs):
            calls.append(env.seed)
            return run_episode(env, transport=honest_transport(), **kwargs)
        def fail_final_write(path, value):
            if Path(path).name == "episode-01-seed-1-target-0.8.json" and value.get("status") == "completed":
                raise OSError("Deliberate local disk-write failure.")
            return write_json(path, value)
        with patch("benchmark_integrity.main_experiment.write_json", side_effect=fail_final_write):
            manifest = self.execute(mocked)
        self.assertEqual(calls, [1])
        self.assertEqual(manifest["status"], "stopped")
        self.assertTrue(all(entry["status"] == "not_started" for entry in manifest["episodes"][1:]))
        saved = read_json(self.output / manifest["episodes"][0]["file"])
        self.assertNotEqual(saved["status"], "completed")
        self.assertTrue(saved["turns"])
        self.assertEqual(saved["source_sha256"], self.plan["source_sha256"])

    def test_nonempty_output_cannot_replace_or_resume(self):
        self.output.mkdir()
        preserved = self.output / "old-attempt.json"
        preserved.write_text("preserve", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "never overwritten"):
            self.execute(lambda *args, **kwargs: self.fail("Must not start."))
        self.assertEqual(preserved.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
