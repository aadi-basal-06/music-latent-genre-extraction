import numpy as np
from music_genre.data.dataset import GenreDataset
from music_genre.data.splits import track_level_split

def test_dataset_and_track_isolation():
    dataset=GenreDataset(np.zeros((4,500)),np.array([0,1,2,3])); assert dataset[0][1].item() == 0
    metadata=[{'track_id':f'{label}-{track}','label':label} for label in range(4) for track in range(8)]
    splits=track_level_split(metadata,seed=1); sets=[set(x) for x in splits.values()]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
