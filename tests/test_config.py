import unittest
from dataclasses import replace
from rasi.config import RASIConfig, ModelConfig, preset_config


class ConfigTests(unittest.TestCase):
    def test_presets_validate(self):
        for name in ("linear-cpu", "linear-native", "gru-cpu", "gru-cuda"):
            self.assertIsInstance(preset_config(name), RASIConfig)

    def test_linear_cuda_rejected(self):
        with self.assertRaises(ValueError):
            ModelConfig(kind="linear", device="cuda:0")

    def test_rasi_constants_are_enforced(self):
        with self.assertRaises(ValueError):
            replace(RASIConfig(), rejection_patience=9)
        with self.assertRaises(ValueError):
            replace(RASIConfig(), verifier_z=2.0)


if __name__ == "__main__":
    unittest.main()
