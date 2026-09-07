import tempfile
import unittest
from pathlib import Path

from scripts.replay_crosswalk_autonomy import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_YOLOE_MODEL,
    _normalized_class_names,
    _resolve_model,
)


class CrosswalkReplayCliTests(unittest.TestCase):
    def test_default_yoloe_model_can_be_downloaded_by_name(self):
        reference, uses_yoloe = _resolve_model(DEFAULT_YOLOE_MODEL)

        self.assertEqual(reference, DEFAULT_YOLOE_MODEL)
        self.assertTrue(uses_yoloe)

    def test_missing_custom_model_fails_before_inference(self):
        with self.assertRaisesRegex(ValueError, "model not found"):
            _resolve_model("weights/missing-best.pt")

    def test_local_custom_model_remains_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "best.pt"
            model.touch()

            reference, uses_yoloe = _resolve_model(str(model))

        self.assertEqual(reference, str(model))
        self.assertFalse(uses_yoloe)

    def test_default_prompts_include_common_crosswalk_aliases(self):
        names = _normalized_class_names(None)

        self.assertEqual(names, DEFAULT_CLASS_NAMES)

    def test_class_names_are_trimmed_normalized_and_deduplicated(self):
        names = _normalized_class_names([" Crosswalk ", "crosswalk", "ZEBRA CROSSING"])

        self.assertEqual(names, ("crosswalk", "zebra crossing"))

    def test_empty_class_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            _normalized_class_names([" "])


if __name__ == "__main__":
    unittest.main()
