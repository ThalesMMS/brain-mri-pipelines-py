import unittest


from brain_mri.scripts.generate_article_tables import _generate_pc2_finetune_summary


class TestGenerateArticleTablesPC2(unittest.TestCase):
    def test_pc2_selection_does_not_require_hardcoded_scenario_prefix(self) -> None:
        experiments = [
            {
                "scenario": "customprefix_efficientnet_classification_seed42",
                "timestamp": "2025-12-14 09:35:23",
                "model": "efficientnet_classification",
                "pretrained": True,
                "freeze_backbone_initial": True,
                "freeze_warmup_epochs": 2,
                "unfreeze_epoch": 3,
                "trainable_params_initial": 12,
                "trainable_params_after_unfreeze": 223,
                "test_accuracy": 0.5,
                "test_f1": 0.1,
                "test_confusion_matrix": [[10, 0], [10, 0]],
            }
        ]

        tex, entry_hash = _generate_pc2_finetune_summary(experiments)
        self.assertIn("customprefix", tex)
        self.assertIsInstance(entry_hash, str)
        self.assertTrue(len(entry_hash) > 0)


if __name__ == "__main__":
    unittest.main()

