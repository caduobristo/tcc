"""Scientific protocol checks: leakage boundaries, metric correctness, saved heads."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch

from src.training.consolidation import (SelectionData, class_weights, fit_candidate,
    fit_preprocessing, metrics_from_cm, rank_finalists, validate_plan, evaluate)
from src.models.consolidated_probe import ConsolidatedProbe

ROOT = Path(__file__).resolve().parents[1]


class ConsolidationTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((ROOT/'configs/linear_probe/consolidation_mono_disc.json').read_text())

    def test_metrics_match_sklearn_including_unpredicted_class(self):
        true = np.tile(np.arange(13),3)
        prediction = true.copy()
        prediction[prediction==7] = 0
        metrics = metrics_from_cm(confusion_matrix(true,prediction,labels=np.arange(13)))
        self.assertAlmostEqual(metrics['accuracy'],accuracy_score(true,prediction))
        self.assertAlmostEqual(metrics['macro_f1'],f1_score(true,prediction,labels=np.arange(13),average='macro',zero_division=0))
        self.assertEqual(metrics['per_class']['OD1']['recall'],0.)

    def test_scaler_fits_training_and_preserves_constant_columns(self):
        train = torch.tensor([[1.,9.],[3.,9.]])
        mean,scale = fit_preprocessing(train,'train_standardize')
        self.assertTrue(torch.equal(mean,torch.tensor([2.,9.])))
        self.assertTrue(torch.allclose(((train-mean)/scale).mean(0),torch.zeros(2)))
        self.assertTrue(bool(torch.isfinite((torch.tensor([[100.,20.]])-mean)/scale).all()))
        self.assertEqual(float(scale[0]),1.)

    def test_weights_use_training_counts_and_fail_on_missing_class(self):
        labels = torch.tensor([0,0,0,1])
        weights = class_weights(labels,'train_balanced',classes=2)
        self.assertTrue(torch.allclose(weights,torch.tensor([2/3,2.])))
        with self.assertRaises(ValueError):
            class_weights(labels,'train_balanced',classes=3)

    def test_ranking_uses_validation_and_rejects_test_exposure(self):
        groups = [{'candidate_id':'a','runs':[{'best_validation_macro_f1':.7,'test_evaluated':False}]},
                  {'candidate_id':'b','runs':[{'best_validation_macro_f1':.8,'test_evaluated':False}]}]
        self.assertEqual(rank_finalists(groups)[0]['candidate_id'],'b')
        groups[0]['runs'][0]['test_evaluated'] = True
        with self.assertRaises(ValueError):
            rank_finalists(groups)

    def test_plan_rejects_fine_tuning_and_shared_confirmation_seeds(self):
        validate_plan(self.plan)
        self.plan['train_encoder'] = True
        with self.assertRaises(ValueError):
            validate_plan(self.plan)
        self.plan['train_encoder'] = False
        self.plan['final_seeds'][0] = 42
        with self.assertRaises(ValueError):
            validate_plan(self.plan)

    def test_training_gate_and_reloaded_best_checkpoint(self):
        torch.set_num_threads(2)
        y = torch.arange(13).repeat(2)
        x = torch.nn.functional.one_hot(y,num_classes=13).float()
        data = SelectionData(x,y,x[:13],y[:13],{'identity':{'synthetic':True},'sha256':'synthetic'},'synthetic')
        candidate = {'normalization':'train_standardize','class_weighting':'train_balanced','batch_size':13}
        self.plan.update(max_epochs=3,early_stopping_patience=3,learning_rate=.1)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)/'probe'
            with self.assertRaises(RuntimeError):
                fit_candidate(data,candidate,42,self.plan,folder)
            self.assertFalse(folder.exists())
            record = fit_candidate(data,candidate,42,self.plan,folder,allow_training=True)
            head = torch.nn.Linear(13,13)
            head.load_state_dict(torch.load(folder/'best_head.pt',weights_only=True))
            prep = torch.load(folder/'preprocessing.pt',weights_only=True)
            metrics,_ = evaluate(head,(data.validation_x-prep['mean'])/prep['scale'],data.validation_y)
            self.assertEqual(metrics['macro_f1'],record['best_validation_macro_f1'])
            self.assertFalse(record['test_evaluated'])
            self.assertEqual(record['optimizer_updates'],6)
            self.assertFalse((folder/'test_metrics.json').exists())

    def test_inference_applies_saved_scaler_and_freezes_encoder(self):
        encoder = torch.nn.Sequential(torch.nn.BatchNorm1d(2),torch.nn.Linear(2,2))
        head = torch.nn.Linear(2,13)
        mean,scale = torch.tensor([1.,2.]),torch.tensor([2.,4.])
        probe = ConsolidatedProbe(encoder,head,mean,scale).train()
        self.assertFalse(encoder.training)
        self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))
        x = torch.tensor([[5.,10.]])
        self.assertTrue(torch.equal(probe.forward_embeddings(x),head(torch.tensor([[2.,2.]]))))


if __name__ == '__main__':
    unittest.main()
