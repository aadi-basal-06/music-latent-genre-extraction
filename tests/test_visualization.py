import numpy as np
from music_genre.visualization.pca import plot_pca
def test_pca_plot(tmp_path):
    reduced=plot_pca(np.random.default_rng(1).normal(size=(12,4)),np.array([0,1,2,3]*3),tmp_path/'plot.png','test'); assert reduced.shape==(12,2) and (tmp_path/'plot.png').exists()
