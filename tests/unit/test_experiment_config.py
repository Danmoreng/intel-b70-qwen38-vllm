import json
import unittest

from scripts.experiment_config import ExperimentConfig, from_environment


class ExperimentConfigTests(unittest.TestCase):
    def test_baseline_is_q128_mtp4_196k(self):
        config = ExperimentConfig()
        args = config.server_args()
        self.assertEqual(args[args.index("--max-model-len") + 1], "200704")
        self.assertEqual(args[args.index("--max-num-batched-tokens") + 1], "4096")
        self.assertEqual(args[args.index("--quantization") + 1], "gptq")
        self.assertEqual(
            json.loads(args[args.index("--speculative-config") + 1]),
            {"method": "mtp", "num_speculative_tokens": 4},
        )

    def test_zero_speculation_omits_entire_option(self):
        args = ExperimentConfig(speculative_tokens=0).server_args()
        self.assertNotIn("--speculative-config", args)

    def test_alternate_quantization_is_an_honest_dry_run(self):
        config = from_environment({"QUANTIZATION": "b70_trellis_h128_v1"})
        args = config.server_args()
        self.assertEqual(args[args.index("--quantization") + 1], "b70_trellis_h128_v1")

    def test_invalid_values_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            ExperimentConfig(speculative_tokens=-1).server_args()
        with self.assertRaisesRegex(ValueError, "positive"):
            ExperimentConfig(max_num_batched_tokens=0).server_args()


if __name__ == "__main__":
    unittest.main()
