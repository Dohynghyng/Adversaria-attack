import cv2

from utils import load_config
import os
import attack
import load_model
import load_data

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

# 원본 prediction
for test_model in model_list:
    model = load_model.load_model(test_model)
    IoU = data_loader.val(model, test_model)
    print(IoU)

# adv prediction
for test_model in model_list:
    model = load_model.load_model(test_model)
    IoU = data_loader.val(model, test_model)
    print(IoU)

