"""Reusable Deep Softmax Autoencoder audio inference."""
from pathlib import Path
import numpy as np
import torch
from music_genre import ID_TO_LABEL
from music_genre.data.audio_loader import load_audio
from music_genre.data.segmentation import segment_audio
from music_genre.data.preprocessing import pool_to_features
from music_genre.models.deep_softmax_autoencoder import DeepSoftmaxAutoencoder
from music_genre.utils.checkpoint import load_checkpoint
from music_genre.utils.device import select_device

def predict_audio(audio_path: str | Path, checkpoint_path: str | Path, device_name: str = 'auto') -> dict:
    """Predict a track by mean-logit aggregation; returned scores are uncalibrated softmax values."""
    device=select_device(device_name); checkpoint=torch.load(checkpoint_path, map_location=device, weights_only=False); cfg=checkpoint['config'].get('model', {})
    model=DeepSoftmaxAutoencoder(**{k: cfg[k] for k in ('input_dim','latent_dim','num_classes','dropout') if k in cfg}).to(device); load_checkpoint(checkpoint_path, model, map_location=device); model.eval()
    segments=segment_audio(load_audio(audio_path))
    if not segments: raise ValueError('Audio contains no complete one-second segments.')
    features=torch.tensor(np.stack([pool_to_features(x) for x in segments]), dtype=torch.float32, device=device)
    with torch.no_grad(): _, logits, embeddings=model(features); mean_logits=logits.mean(0); probabilities=torch.softmax(mean_logits, dim=0).cpu().numpy(); embedding=embeddings.mean(0).cpu().numpy()
    predicted=int(probabilities.argmax())
    return {'genre': ID_TO_LABEL[predicted], 'scores': {ID_TO_LABEL[i]: float(probabilities[i]) for i in range(4)}, 'embedding': embedding.tolist(), 'segment_predictions': [ID_TO_LABEL[int(i)] for i in logits.argmax(1).cpu().tolist()]}
