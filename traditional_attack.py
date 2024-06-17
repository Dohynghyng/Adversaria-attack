import cv2

from utils import load_config, load_data
import os
import attack
from utils import load_model

if not os.path.isdir('./runs'):
    os.mkdir('./runs')

config = load_config.loadConfig()

config['save_path'] = './runs/' + config['save_path'] + '/'
os.mkdir(config['save_path'])

data_loader = load_data.dataLoader()

source_model_list = ['resnet50', 'resnet101']
model_list = ['fcn_resnet50', 'fcn_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101']

for model_name in model_list:
    model = load_model.load_model(model_name)
    path = config['save_path'] + model_name + '/'
    os.mkdir(path)
    for idx in range(len(data_loader.f_names)):
        X, y, f_name = data_loader.traditional_batch_load(idx)
        adv_ex = attack.traditional_ifgsm(model,X,y, config)
        cv2.imwrite(path+f_name+'.jpg',cv2.cvtColor(adv_ex,cv2.COLOR_RGB2BGR))
        if (idx%100) == 0:
            print(f"[{model_name}] {idx} / {len(data_loader.f_names)}")

# adv prediction
for source_model in model_list:
    data_loader.apply_traditional_adv(source_model)
    for model_name in model_list:
        model = load_model.load_model(model_name)
        IoU = data_loader.val(model, model_name,avd_name=source_model)
        with open(f"{config['save_path']}IoU.txt", "a") as f:
            f.write(f"{source_model} > {model_name} : {IoU}\n")
