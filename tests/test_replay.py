"""Offline evidence verification, including tampered saved snapshots."""
import copy
import unittest

from benchmark_integrity.__main__ import verify_replay
from benchmark_integrity.controls import make_control


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.artifact = make_control("honest")

    def test_complete_artifact_and_bare_environment_replay(self):
        self.assertEqual(verify_replay(self.artifact), self.artifact["grade"])
        self.assertEqual(verify_replay(self.artifact["environment"]), self.artifact["grade"])

    def test_tampered_evaluation_state_rejected_even_with_unchanged_tool_results(self):
        altered = copy.deepcopy(self.artifact)
        altered["environment"]["evaluations"][0]["threshold"] = 99
        with self.assertRaisesRegex(ValueError, "evaluations"):
            verify_replay(altered)

    def test_tampered_evaluation_population_rejected(self):
        altered = copy.deepcopy(self.artifact)
        altered["environment"]["evaluations"][0]["row_ids"].pop()
        with self.assertRaisesRegex(ValueError, "evaluations"):
            verify_replay(altered)

    def test_tampered_action_metadata_rejected(self):
        altered = copy.deepcopy(self.artifact)
        altered["environment"]["actions"][0]["unauthorized_mutation"] = True
        with self.assertRaisesRegex(ValueError, "actions"):
            verify_replay(altered)

    def test_tampered_original_dataset_rejected(self):
        altered = copy.deepcopy(self.artifact)
        altered["environment"]["dataset"][0]["y"] ^= 1
        with self.assertRaisesRegex(ValueError, "dataset"):
            verify_replay(altered)

    def test_changed_json_types_do_not_pass_python_boolean_number_equality(self):
        for location in ["experiment", "tool_result", "envelope_grade"]:
            with self.subTest(location=location):
                altered = copy.deepcopy(self.artifact)
                if location == "experiment":
                    altered["environment"]["evaluations"][0]["population"]["is_original"] = 1
                elif location == "tool_result":
                    altered["environment"]["actions"][0]["result"]["ok"] = 1
                else:
                    altered["grade"]["submitted_mismatch"] = 0
                with self.assertRaises(ValueError):
                    verify_replay(altered)

    def test_tampered_tool_result_rejected(self):
        altered = copy.deepcopy(self.artifact)
        altered["environment"]["actions"][0]["result"]["ok"] = False
        with self.assertRaisesRegex(ValueError, "action 1"):
            verify_replay(altered)

    def test_conflicting_envelope_grade_rejected(self):
        altered = copy.deepcopy(self.artifact)
        altered["grade"]["reported_score"] = -1
        with self.assertRaisesRegex(ValueError, "envelope grade"):
            verify_replay(altered)

    def test_missing_inner_or_outer_grade_is_not_silently_accepted(self):
        for location in ["environment", None]:
            with self.subTest(location=location):
                altered = copy.deepcopy(self.artifact)
                del (altered[location] if location else altered)["grade"]
                with self.assertRaises(ValueError):
                    verify_replay(altered)

    def test_historical_schema_requires_preserved_source(self):
        altered = copy.deepcopy(self.artifact)
        altered["environment"]["schema_version"] = "historical-prototype"
        with self.assertRaisesRegex(ValueError, "preserved source"):
            verify_replay(altered)

    def test_malformed_artifact_has_clear_failure(self):
        for altered in [None, {"environment": []}, {}]:
            with self.subTest(artifact=altered), self.assertRaises(ValueError):
                verify_replay(altered)


if __name__ == "__main__":
    unittest.main()
