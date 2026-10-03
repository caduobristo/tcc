"""Check TS9 removal, class remapping and compatibility with the original protocol."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from src.data.transfer import EFFECTS
from src.models.consolidated_probe import load_consolidated_probe
from src.training.consolidation import (classes_for_plan, dataset_view, experiment_suffix,
    metrics_from_cm, SelectionData, fit_candidate, evaluate, validate_plan)
from scripts.report_transfer_no_ts9 import restricted_original_metrics

ROOT = Path(__file__).resolve().parents[1]


class AblationTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((ROOT/'configs/linear_probe/consolidation_mono_disc.json').read_text())
        self.plan['excluded_effects'] = ['TS9']
        self.classes = classes_for_plan(self.plan)

    def rows(self):
        return [{'effect': effect, 'label': str(EFFECTS.index(effect)), 'split': split, 'source_group': split}
                for split in ('train', 'validation', 'test') for effect in EFFECTS]

    def test_filtered_indices_preserve_partitions_and_remap_vtb(self):
        rows = self.rows()
        indices, view = dataset_view(rows, self.classes)
        self.assertEqual(len(indices), 36)
        self.assertEqual(view['excluded_samples'], 3)
        self.assertEqual(view['split_counts'], dict(train=12, validation=12, test=12))
        self.assertTrue(all(rows[i]['effect'] != 'TS9' for i in indices))
        self.assertEqual(self.classes.index('VTB'), 11)
        self.assertEqual([rows[i]['source_group'] for i in indices[:12]], ['train']*12)
        self.assertEqual(view, dataset_view(rows, self.classes)[1])

    def test_original_mapping_and_source_leakage_are_rejected_even_on_excluded_rows(self):
        rows = self.rows()
        rows[11]['source_group'] = 'validation'
        with self.assertRaises(ValueError):
            dataset_view(rows, self.classes)
        rows = self.rows()
        rows[-1]['label'] = '11'
        with self.assertRaises(ValueError):
            dataset_view(rows, self.classes)

    def test_invalid_class_exclusions_and_original_default(self):
        validate_plan(self.plan)
        self.assertEqual(experiment_suffix(self.plan), 'no_ts9')
        self.plan['excluded_effects'] = ['TS9', 'TS9']
        with self.assertRaises(ValueError):
            validate_plan(self.plan)
        self.plan['excluded_effects'] = ['unknown']
        with self.assertRaises(ValueError):
            validate_plan(self.plan)
        del self.plan['excluded_effects']
        self.assertEqual(classes_for_plan(self.plan), EFFECTS)
        self.assertEqual(experiment_suffix(self.plan), '')

    def test_twelve_class_metrics_include_unpredicted_class(self):
        truth = np.tile(np.arange(12), 3)
        prediction = truth.copy()
        prediction[prediction == 11] = 0
        metrics = metrics_from_cm(confusion_matrix(truth, prediction, labels=np.arange(12)), self.classes)
        self.assertAlmostEqual(metrics['macro_f1'], f1_score(truth, prediction, average='macro', zero_division=0))
        self.assertNotIn('TS9', metrics['per_class'])
        self.assertEqual(metrics['per_class']['VTB']['recall'], 0)
        with self.assertRaises(ValueError):
            metrics_from_cm(np.eye(13, dtype=int), self.classes)

    def test_original_ts9_prediction_still_counts_as_error_on_retained_wavs(self):
        metrics = restricted_original_metrics(np.asarray([0, 12]), np.asarray([11, 12]), ('808', 'VTB'))
        self.assertEqual(metrics['accuracy'], .5)
        self.assertEqual(metrics['macro_f1'], .5)
        self.assertEqual(metrics['per_class']['808']['recall'], 0.)
        with self.assertRaises(ValueError):
            restricted_original_metrics(np.asarray([0, 11]), np.asarray([0, 11]), ('808', 'VTB'))

    def test_consultation_notebook_cannot_launch_training_by_default(self):
        notebook = json.loads((ROOT/'notebooks/transfer_learning_no_ts9.ipynb').read_text(encoding='utf-8'))
        sources = '\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type']=='code')
        self.assertIn('RUN_ABLATION = False', sources)
        self.assertNotIn('RUN_ABLATION = True', sources)
        self.assertTrue(all(not cell.get('outputs') and cell.get('execution_count') is None
                            for cell in notebook['cells'] if cell['cell_type']=='code'))

    def test_saved_twelve_output_head_and_scaler_load_with_correct_labels(self):
        torch.set_num_threads(2)
        y = torch.arange(12).repeat(2)
        x = torch.nn.functional.one_hot(y, num_classes=768).float()
        cache = {'identity': {'encoder_precision': 'float32'}, 'sha256': 'synthetic'}
        data = SelectionData(x, y, x[:12], y[:12], cache, 'passt', self.classes)
        candidate = dict(normalization='train_standardize', class_weighting='train_balanced', batch_size=12)
        self.plan.update(max_epochs=3, early_stopping_patience=3, learning_rate=.1)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)/'head'
            record = fit_candidate(data, candidate, 42, self.plan, folder, allow_training=True)
            self.assertEqual(record['trainable_parameters'], 9228)
            with patch('src.models.consolidated_probe.load_frozen_encoder', return_value=torch.nn.Identity()):
                probe = load_consolidated_probe(folder, device='cpu')
            self.assertEqual(probe.classes, self.classes)
            self.assertEqual(probe(x[:12]).shape, (12, 12))
            metrics, _ = evaluate(probe.head, (x[:12]-probe.mean)/probe.scale, y[:12], classes=probe.classes)
            self.assertEqual(metrics['macro_f1'], record['best_validation_macro_f1'])
            self.assertFalse((folder/'test_metrics.json').exists())

    def test_all_four_plans_change_only_the_class_exclusion_and_exposure_flag(self):
        for scenario in ('mono_disc', 'mono_cont', 'poly_disc', 'poly_cont'):
            original = json.loads((ROOT/f'configs/linear_probe/consolidation_{scenario}.json').read_text())
            ablation = json.loads((ROOT/f'configs/linear_probe/consolidation_no_ts9_{scenario}.json').read_text())
            validate_plan(ablation)
            self.assertEqual(ablation.pop('excluded_effects'), ['TS9'])
            ablation['historical_test_already_inspected'] = original['historical_test_already_inspected']
            self.assertEqual(ablation, original)


if __name__ == '__main__':
    unittest.main()
