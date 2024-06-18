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

for source_model in source_model_list:
    model = load_model.load_encoder(source_model)
    path = config['save_path'] + source_model + '/'
    os.mkdir(path)
    for idx in range(len(data_loader.f_names)):
        X, f_name = data_loader.adv_batch_load(idx)
        adv_ex = attack.ifgsm(model,X,config)
        cv2.imwrite(path+f_name+'.jpg',cv2.cvtColor(adv_ex,cv2.COLOR_RGB2BGR))
        if (idx%100) == 0:
            print(f"[{source_model}] {idx} / {len(data_loader.f_names)}")

# normal prediction
for test_model in model_list:
    if not os.path.isdir('./runs/' + test_model + '/'):
        os.mkdir('./runs/' + test_model + '/')
        model = load_model.load_model(test_model)
        IoU = data_loader.val(model, test_model)
        try:
            with open("/runs/IoU.txt", "a") as f:
                f.write(f"{test_model} : {IoU}\n")
        except:
            with open("/runs/IoU.txt", "w") as f:
                f.write(f"{test_model} : {IoU}\n")
        print(f"{test_model} : {IoU}")

# adv prediction
for source_model in source_model_list:
    data_loader.apply_adv(source_model)
    for test_model in model_list:
        model = load_model.load_model(test_model)
        IoU = data_loader.val(model, test_model,avd_name=source_model)
        with open(f"{config['save_path']}IoU.txt", "a") as f:
            f.write(f"{source_model}_{test_model} : {IoU}\n")
        print(f"{source_model}_{test_model} : {IoU}")
        print(f"{source_model}_{test_model} : {IoU}")

    with open(f"{config['save_path']}IoU.txt", "a") as f:
        f.write("\n")
