import torch
import torch.nn as nn

from torchvision import models

def load_model(model_name):
    model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
    model.cuda()
    model.eval()
    return model

def load_encoder(model_name):
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=True)
    model.avgpool = nn.Identity()
    model.fc = nn.Identity()
    model.cuda()
    model.eval()
    return model
