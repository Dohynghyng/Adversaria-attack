from utils import load_config

import load_model
import load_data

config = load_config.loadConfig()

data_loader = load_data.dataLoader()
source_model_list = ['fcn_resnet50', 'fcn_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101']


for source_model in source_model_list:
    source_model = load_model.loadModel(source_model)

    # attack 코드

# IoU 계산
