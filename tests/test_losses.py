import torch
from music_genre.models import DeepSoftmaxAutoencoder
from music_genre.models.losses import hybrid_loss, source_joint_loss

def test_hybrid_loss_and_gradients():
    model=DeepSoftmaxAutoencoder(); x=torch.randn(4,500); y=torch.tensor([0,1,2,3]); reconstruction, logits, _=model(x); total, rec, cls=hybrid_loss(reconstruction,logits,x,y,.9)
    assert torch.allclose(total, .9*rec+.1*cls); total.backward()
    assert any(p.grad is not None for p in model.encoder.parameters())
    assert any(p.grad is not None for p in model.decoder.parameters())
    assert any(p.grad is not None for p in model.classifier.parameters())

def test_shared_encoder_receives_each_branch_gradient():
    model=DeepSoftmaxAutoencoder(dropout=0); x=torch.randn(4,500); y=torch.tensor([0,1,2,3])
    reconstruction, logits, _=model(x)
    reconstruction_loss=torch.nn.functional.mse_loss(reconstruction,x)
    reconstruction_grad=torch.autograd.grad(reconstruction_loss, tuple(model.encoder.parameters()), retain_graph=True)
    classification_loss=torch.nn.functional.cross_entropy(logits,y)
    classification_grad=torch.autograd.grad(classification_loss, tuple(model.encoder.parameters()))
    assert any(gradient.abs().sum() > 0 for gradient in reconstruction_grad)
    assert any(gradient.abs().sum() > 0 for gradient in classification_grad)

def test_source_joint_loss_matches_original_weighting():
    reconstruction=torch.zeros(2, 2); inputs=torch.ones(2, 2); logits=torch.tensor([[2.,0.,0.,0.],[0.,2.,0.,0.]])
    targets=torch.tensor([0,1]); total, rec, cls=source_joint_loss(reconstruction, logits, inputs, targets, .1)
    assert torch.allclose(total, rec + .1 * cls)
