import os
import unittest
from rasi.config import ModelConfig
from rasi.diagnostics import gpu_test, model_smoke, system_report


class DiagnosticTests(unittest.TestCase):
    def test_system_report(self):
        report = system_report()
        self.assertIn("python", report)
        self.assertIn("numpy", report)
        self.assertIn("torch", report)

    def test_linear_smoke(self):
        report = model_smoke(ModelConfig(kind="linear", backend="numpy"))
        self.assertTrue(report["passed"])
        self.assertTrue(report["checkpoint_roundtrip"])

    @unittest.skipUnless(os.environ.get("RASI_TEST_CUDA") == "1", "set RASI_TEST_CUDA=1 to require CUDA")
    def test_cuda(self):
        report = gpu_test(os.environ.get("RASI_TEST_DEVICE", "cuda:0"), matrix_size=64, repeats=1)
        self.assertTrue(report["passed"], report)


if __name__ == "__main__":
    unittest.main()
