import unittest
import numpy as np
from src.utils.standardization import LogZScoreStandardizer

class TestUtils(unittest.TestCase):
    def test_log_zscore(self):
        mean_log = 1.0
        std_log = 0.5
        std = LogZScoreStandardizer(mean_log=mean_log, std_log=std_log)
        
        # Inverse: z=0 -> x_log=1 -> x=e^1 - 1
        z = np.array([0.0])
        x = std.inverse_transform(z)
        expected = np.expm1(1.0)
        self.assertAlmostEqual(x[0], expected)
        
        # Forward
        z_rec = std.transform(x)
        self.assertAlmostEqual(z_rec[0], 0.0)

if __name__ == '__main__':
    unittest.main()
