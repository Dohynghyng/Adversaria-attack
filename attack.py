import torch
import torch.nn as nn


mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
mean, std = torch.Tensor(mean).cuda(), torch.Tensor(std).cuda()

def ifgsm_DH(model, X, config):
    model.eval()
    model.cuda()

    X_pert = X.clone()
    X = X.clone()
    X_pert.requires_grad = True
    for i in range(config['n_iter']):
        output_perturbed = model(X_pert)
        y_used = torch.zeros_like(output_perturbed) - config['delta']
        y_used = torch.FloatTensor(y_used.cpu()).cuda()

        loss = nn.MSELoss()(output_perturbed, y_used)
        loss.backward()

        pert = 1 * config['lr'] * X_pert.grad.detach().sign()
        X_pert = update_adv(X, X_pert, pert, config['eps'])
        X_pert.requires_grad = True

    return X_pert.clone().detach()


def update_adv(X, X_pert, pert, epsilon):
    X = X.clone().detach()
    X_pert = X_pert.clone().detach()
    X_pert = X_pert + pert
    noise = X_pert - X
    noise = torch.permute(noise,(0,2,3,1))
    noise = torch.clamp(noise, -epsilon/std, epsilon/std)
    noise = torch.permute(noise,(0,3,1,2))
    X_pert = X + noise

    X_pert = torch.permute(X_pert,(0,2,3,1))
    X_pert = torch.clamp(X_pert, min=-mean/std, max=(1-mean)/std)
    X_pert = torch.permute(X_pert,(0,3,1,2))
    return X_pert.clone().detach()