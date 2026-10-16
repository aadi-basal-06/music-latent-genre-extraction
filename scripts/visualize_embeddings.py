#!/usr/bin/env python3
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from music_genre.data.dataset import load_processed_split
from music_genre.models.deep_softmax_autoencoder import DeepSoftmaxAutoencoder
from music_genre.utils.checkpoint import load_checkpoint
from music_genre.utils.device import select_device
from music_genre.visualization.pca import plot_pca
parser=argparse.ArgumentParser(); parser.add_argument('--checkpoint',required=True); parser.add_argument('--processed-dir',default='data/processed'); parser.add_argument('--splits-path',default='data/splits/splits.json'); parser.add_argument('--output-dir',default='outputs/figures'); args=parser.parse_args(); device=select_device(); ck=torch.load(args.checkpoint,map_location=device,weights_only=False); model=DeepSoftmaxAutoencoder(**ck['config']['model']).to(device); load_checkpoint(args.checkpoint,model,map_location=device); dataset=load_processed_split(args.processed_dir,args.splits_path,'test'); x=dataset.features.numpy(); y=dataset.labels.numpy(); model.eval()
with torch.no_grad(): _, logits, embeddings=model(torch.tensor(x,device=device))
plot_pca(x,y,f'{args.output_dir}/raw_inputs_pca.png','PCA: raw 500-D waveform features'); plot_pca(embeddings.cpu().numpy(),y,f'{args.output_dir}/deep_embeddings_pca.png','PCA: deep autoencoder embeddings'); plot_pca(logits.cpu().numpy(),y,f'{args.output_dir}/deep_logits_pca.png','PCA: deep autoencoder logits')
