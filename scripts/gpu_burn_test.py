"""Tests for the GPU burn utility."""

import unittest

from scripts.gpu_burn import resolve_target_bytes


class ResolveTargetBytesTest(unittest.TestCase):
    """Verify VRAM target calculations stay within safe bounds."""

    def test_uses_requested_ratio_within_range(self) -> None:
        """Return the expected byte count for an in-range ratio."""
        self.assertEqual(resolve_target_bytes(1000, 0.5), 500)

    def test_clamps_ratio_to_safe_limits(self) -> None:
        """Clamp extreme ratios before converting to bytes."""
        self.assertEqual(resolve_target_bytes(1000, 0.0), 50)
        self.assertEqual(resolve_target_bytes(1000, 1.0), 950)


if __name__ == "__main__":
    unittest.main()
