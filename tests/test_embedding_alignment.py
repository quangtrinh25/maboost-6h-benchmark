"""Stage-2 training embeddings must stay row-aligned with labels and tabular designs."""
from pathlib import Path
import sys, tempfile, unittest
from unittest import mock
import numpy as np
import torch.nn as nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmark/src'))
sys.path.insert(0,str(ROOT))
import v13_training as T
from v13_feature_registry import DYNAMIC_FEATURES, STATIC_FEATURES
from v13_guard import WORKSPACE_ROOT


def _arrays(n=96, steps=8):
    rng=np.random.default_rng(0)
    mortality=(rng.random(n)<0.5).astype(np.int8)
    values=rng.normal(size=(n,steps,len(DYNAMIC_FEATURES))).astype(np.float32)
    values[:,0,0]=mortality  # the stub embedding exposes the row's own label
    return {
        'values':values,
        'mask':np.ones((n,steps,len(DYNAMIC_FEATURES)),np.uint8),
        'tau':np.full((n,steps,len(DYNAMIC_FEATURES)),60.0,np.float32),
        'elapsed':np.tile(np.linspace(0.5,6,steps,dtype=np.float32),(n,1)),
        'static':rng.normal(size=(n,len(STATIC_FEATURES))).astype(np.float32),
        'static_mask':np.ones((n,len(STATIC_FEATURES)),np.float32),
        'mortality':mortality,
        'los':(np.arange(n,dtype=np.float32)%7)+1.0,
        'los_days':(np.arange(n,dtype=np.float32)%7)+1.0,
    }


class _StubEncoder(nn.Module):
    def extract_features(self, values, tau, mask):
        return values[:,0,:1].repeat(1,4), None


class _StubMamba(nn.Module):
    def __init__(self, **_unused):
        super().__init__()
        self.head=nn.Linear(1,2)
        self.los=nn.Linear(1,1)
        self.encoder=_StubEncoder()
    def forward(self, values, tau, mask, static):
        feature=values[:,0,:1]
        return self.head(feature), self.los(feature).squeeze(-1)


class _Captured(Exception):
    pass


class EmbeddingAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.arrays=_arrays()
        n=len(self.arrays['mortality'])
        self.indices={'train':np.arange(0,64),'validation':np.arange(64,80),'test':np.arange(80,n)}
        self.normalization=T.fit_normalization_chunked(self.arrays,self.indices['train'])

    def test_inference_loaders_preserve_index_order(self):
        loaders=T.make_loaders(self.arrays,self.indices,self.normalization,batch_size=8,seed=42,shuffle_train=False)
        for split,rows in self.indices.items():
            observed=np.concatenate([batch[4].numpy() for batch in loaders[split]])
            np.testing.assert_array_equal(observed,self.arrays['mortality'][rows])

    def test_training_loader_still_shuffles(self):
        loaders=T.make_loaders(self.arrays,self.indices,self.normalization,batch_size=8,seed=42)
        rows=np.concatenate([batch[5].numpy() for batch in loaders['train']])
        self.assertFalse(np.array_equal(rows,self.arrays['los'][self.indices['train']]))

    def _stage2_inputs(self, model_name):
        captured={}
        def capture(train_features, validation_features, train_mortality, validation_mortality, *args, **kwargs):
            captured.update(train=np.asarray(train_features),labels=np.asarray(train_mortality),
                            validation=np.asarray(validation_features),validation_labels=np.asarray(validation_mortality))
            raise _Captured()
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as directory, \
             mock.patch.object(T,'instantiate_maboost',_StubMamba), \
             mock.patch.object(T,'_fit_xgb_pair',capture):
            with self.assertRaises(_Captured):
                T.train_xgb_or_mamba_primary(model_name,self.arrays,self.indices,self.normalization,seed=42,
                    max_epochs=1,patience=1,n_estimators=2,device='cpu',job_root=Path(directory),
                    training_config={'mamba':{'longitudinal_training':False}},bounded=True)
        return captured

    def test_stage2_train_embeddings_align_with_labels(self):
        for model_name in ('Mamba-z-only','Hybrid-plain','MABOOST'):
            with self.subTest(model=model_name):
                captured=self._stage2_inputs(model_name)
                np.testing.assert_array_equal(captured['labels'],self.arrays['mortality'][self.indices['train']])
                np.testing.assert_array_equal((captured['train'][:,0]>0).astype(np.int8),captured['labels'])
                np.testing.assert_array_equal((captured['validation'][:,0]>0).astype(np.int8),captured['validation_labels'])

    def test_runtime_guard_rejects_misaligned_loaders(self):
        def shuffled_encoder(arrays, indices, normalization, *, seed, **kwargs):
            loaders=T.make_loaders(arrays,indices,normalization,batch_size=8,seed=seed,shuffle_train=True)
            return _StubMamba(),{},loaders
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as directory, \
             mock.patch.object(T,'train_mamba_encoder',shuffled_encoder):
            with self.assertRaisesRegex(RuntimeError,'not row-aligned'):
                T.train_xgb_or_mamba_primary('MABOOST',self.arrays,self.indices,self.normalization,seed=42,
                    max_epochs=1,patience=1,n_estimators=2,device='cpu',job_root=Path(directory),
                    training_config={'mamba':{}},bounded=True)


if __name__=='__main__':
    unittest.main()
