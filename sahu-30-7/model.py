import torch, torchvision, torchsummary
import os
from torch import nn


shufflenet = torchvision.models.shufflenet_v2_x1_0()

shufflenet.fc = nn.Sequential(
    nn.Linear(1024, 512), # have to calculate the input size
    nn.PReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.PReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 8),
    nn.LogSoftmax(dim=1)
)

if torch.cuda.is_available():
    shufflenet = shufflenet.cuda()
