import numpy as np
import soundfile as sf
import torch
from music_genre.models import DeepSoftmaxAutoencoder
from music_genre.utils.checkpoint import save_checkpoint
from music_genre.predict import predict_audio

def test_inference_output_structure(tmp_path):
    audio=tmp_path/'tone.wav'; sf.write(audio, np.zeros(22050, dtype=np.float32), 22050)
    model=DeepSoftmaxAutoencoder(); optimizer=torch.optim.Adam(model.parameters())
    config={'model': {'input_dim':500, 'latent_dim':64, 'num_classes':4, 'dropout':.1}}
    checkpoint=tmp_path/'model.pt'; save_checkpoint(checkpoint,model,optimizer,1,0.,config,42)
    result=predict_audio(audio,checkpoint,device_name='cpu')
    assert set(result) == {'genre','scores','embedding','segment_predictions'}
    assert len(result['embedding']) == 64 and len(result['scores']) == 4
