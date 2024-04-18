import torch
from torchvision import models

def load_pre_trained_model(model_name):
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=True)
    model.avgpool = None
    model.fc = None
    return model


def loadModel(model_name):
    """
    model_name : fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101
    """
    model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
    model.cuda()
    model.eval()
    return model
