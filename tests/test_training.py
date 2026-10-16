import torch
from torch.utils.data import DataLoader
from music_genre.data.dataset import GenreDataset
from music_genre.models import TwoLayerClassifier
from music_genre.training import Trainer
from music_genre.utils.checkpoint import load_checkpoint

def test_synthetic_training_and_checkpoint(tmp_path):
    ds=GenreDataset(torch.randn(12,500).numpy(),torch.tensor([0,1,2,3]*3).numpy()); model=TwoLayerClassifier(); trainer=Trainer(model,torch.optim.Adam(model.parameters()),torch.device('cpu'),{'x':1},'baseline'); history=trainer.fit(DataLoader(ds,batch_size=4),DataLoader(ds,batch_size=4),1,tmp_path/'ckpt',tmp_path/'out',42)
    restored=TwoLayerClassifier(); checkpoint=load_checkpoint(tmp_path/'ckpt/best.pt',restored)
    assert len(history)==1 and checkpoint['epoch']==1
