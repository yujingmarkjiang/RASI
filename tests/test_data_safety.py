import unittest
from rasi.data import Contrast, Query, Trajectory


class DataSafetyTests(unittest.TestCase):
    def test_individual_contrast_requires_same_subject(self):
        a = Query("a", "s1", "A", (0.0, 1.0))
        b = Query("b", "s2", "B", (0.0, 1.0))
        with self.assertRaises(ValueError):
            Contrast("c", (a,), (b,), (0.0, 0.0), "u", "randomised_crossover")

    def test_real_trajectory_cannot_claim_synthetic_donors(self):
        q = Query("q", "s", "A", (0.0, 1.0))
        with self.assertRaises(ValueError):
            Trajectory(q, (1.0, 2.0), origin="real", donors=("x",))


if __name__ == "__main__":
    unittest.main()
