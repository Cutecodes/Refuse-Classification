
from CNN import CNN
import torch
from torchvision.models import AlexNet
from torchviz import make_dot
 
x=torch.rand(1,3,64,64)
model=CNN(64,6)
y=model(x)
g=make_dot(y, params=dict(model.named_parameters()))
g.view()

