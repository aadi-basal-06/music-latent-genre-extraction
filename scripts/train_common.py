"""Shared CLI implementation for the three train entry points."""
import argparse
import torch
from torch.utils.data import DataLoader
from music_genre.data.dataset import load_processed_split
from music_genre.models import TwoLayerClassifier, VanillaAutoencoder, DeepSoftmaxAutoencoder
from music_genre.training import Trainer
from music_genre.utils.config import load_config
from music_genre.utils.device import select_device
from music_genre.utils.reproducibility import set_seed

def main(mode: str) -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--config', required=True); parser.add_argument('--resume'); args=parser.parse_args(); cfg=load_config(args.config); set_seed(cfg['seed']); device=select_device(cfg.get('device','auto'))
    train=load_processed_split(cfg['data']['processed_dir'], cfg['data']['splits_path'], 'train'); val=load_processed_split(cfg['data']['processed_dir'], cfg['data']['splits_path'], 'val')
    model={'baseline': TwoLayerClassifier, 'autoencoder': VanillaAutoencoder, 'deep_softmax': DeepSoftmaxAutoencoder}[mode](**cfg['model'])
    optimizer=torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate']); loss_cfg=cfg.get('loss', {}); trainer=Trainer(model, optimizer, device, cfg, {'deep_softmax':'deep_softmax'}.get(mode, mode), loss_cfg.get('gamma',.9), loss_cfg.get('mode','convex'), loss_cfg.get('classification_weight',.1))
    if args.resume:
        from music_genre.utils.checkpoint import load_checkpoint
        checkpoint=load_checkpoint(args.resume, model, optimizer, device); start=checkpoint['epoch']
    else: start=0
    history=trainer.fit(DataLoader(train, batch_size=cfg['training']['batch_size'], shuffle=True), DataLoader(val, batch_size=cfg['training']['batch_size']), cfg['training']['epochs'], cfg['training']['checkpoint_dir'], cfg['training']['output_dir'], cfg['seed'], start)
    from music_genre.visualization.training_curves import plot_training_curves
    plot_training_curves(history, f"{cfg['training']['output_dir']}/training_curve.png")
if __name__ == '__main__': main('baseline')
