import dubhe_sdk

from geek_vision_algo.train import main as model_train
from geek_vision_algo.predict import main as model_predict
from geek_vision_algo.inference import main as model_inference

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split
import glob
import re
import shutil
import traceback
import json
import os
import pandas as pd
import numpy as np
from dataset.pretrain_prepare import main as prepare_data
from dataset import MyDataset
import torch
import geek_vision_algo.model as mm

def load_config(config_file):
    try:
        with open(config_file) as file:
            config =json.load(file)
            return config
    except OSError as e:
        print(e)
        return None

@dubhe_sdk.pipeline()
class Train_Model():
    def __init__(self, ctx:dubhe_sdk.Context):
        config_params = load_config('config/model_train_config.json')
        self.model_name = config_params['model']
        self.evaluation_interval = config_params['evaluation_interval']
        self.epochs = config_params['epochs']
        self.batch_size = config_params['batch_size']
        self.lr = config_params['lr']


    @dubhe_sdk.train()
    def train(self, ctx: dubhe_sdk.Context):
        # get input parameters
        self.data_paths = ctx.get(dubhe_sdk.ContextKey.KEY_DATA_PATHS)
        self.code_list = ctx.get(dubhe_sdk.ContextKey.KEY_CODE_LIST)
        self.pretrained_model_path = ctx.get(dubhe_sdk.ContextKey.KEY_PRETRAINED_MODEL_PATH)
        self.pretrained_epoch = ctx.get(dubhe_sdk.ContextKey.KEY_PRETRAINED_EPOCH)
        self.output_dir = ctx.get(dubhe_sdk.ContextKey.KEY_OUTPUT_DIR)
        self.debug_dir = ctx.get(dubhe_sdk.ContextKey.KEY_TMP_DIR)
        self.split_rate = ctx.get(dubhe_sdk.ContextKey.KEY_SPLIT_RATE)
        self.platform_type = ctx.get(dubhe_sdk.ContextKey.ENV_PLATFORM_TYPE)

        # choose gpu to train
        if (self.platform_type != 1):
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # prepeare dir
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir, exist_ok=True)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        # split train and test
        prepare_data_args = [
            "--data_paths",
            *self.data_paths,
            "--code_list",
            *self.code_list,
            "--val_ratio=%s"%self.split_rate,
            "--debug_dir=%s"%self.debug_dir,
            '--predict_method=train',
        ]
        prepare_data(prepare_data_args)
        df_train = pd.read_csv(os.path.join(self.debug_dir, "train.csv"))
        df_valid = pd.read_csv(os.path.join(self.debug_dir, "val.csv"))
        code_count_dict = dict(zip(*np.unique(df_train["code"].values.tolist(), return_counts=True)))
        code_count_list = sorted(code_count_dict, key=lambda i:i[1], reverse=True)
        code2label = {}
        for idx, code in enumerate(code_count_list):
            code2label[code] = idx

        ctx.log(f'Code list used for training is  {code_count_list}.')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0),
            transforms.ColorJitter(brightness=0, contrast=0, saturation=0,
                                   hue=0),
            transforms.ToTensor(),
            normalize,
        ])
        valid_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = MyDataset(df_train,
                                  code2label,
                                  train_transforms)
        valid_dataset = MyDataset(df_valid,
                                  code2label,
                                  valid_transforms)

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset


        # train model
        ctx.log('model train starting...')
        train_args = [
            '--model_name=%s' % self.model_name,
            '--init_epoch=%d' % self.pretrained_epoch,
            '--epochs=%d' % self.epochs,
            '--batch_size=%d' % self.batch_size,
            '--lr=%f' % self.lr,
            '--evaluation_interval=%d' % self.evaluation_interval,
            '--save_path=%s' % self.output_dir,
            '--progress_init=0.0',
            '--progress_ratio=0.9',
            '--model_optimizer=SGD',
            '--code_list',
            *code_count_list,
            '--model_num=0'
        ]
        ctx.log('model train args \n %s ' % train_args)
        model_train(train_args, self.train_dataset, self.valid_dataset, ctx)  # please design your model train logic in this function.
        ctx.log('model train end.')

        # 2. find best model file
        model_files = glob.glob(os.path.join(self.output_dir, '*'))
        model_prefix = r'model_(\d)+_{}'.format(self.epochs - 1)
        for file in model_files:
            model_name = os.path.basename(file)
            if re.match(model_prefix, model_name, re.M | re.I):

                self.model_file = model_name
                break


        # 3. predict model
        ctx.log('model predict starting...')

        self.output_txt = os.path.join(self.debug_dir, 'pred_result.txt')
        self.output_img = os.path.join(self.debug_dir, 'vis')

        model_path = os.path.join(self.output_dir, self.model_file)
        if model_path:
            model_dict = torch.load(model_path)
            self.state_dict = model_dict['state_dict']

        predict_args = [
            '--model_name=%s' % self.model_name,
            '--output_txt=%s' % self.output_txt,
            '--output_img=%s' % self.output_img,
            '--batch_size=%d' % self.batch_size,
            '--progress_init=0.9',
            '--progress_ratio=0.1',
            '--code_list',
            *code_count_list,
            '--model_num=0'
        ]
        ctx.log('model predict args \n %s' % predict_args)
        model_predict(predict_args, self.valid_dataset, self.state_dict, ctx)
        ctx.log('model predict end.')

@dubhe_sdk.pipeline()
class Predict_Model():
    def __init__(self, ctx:dubhe_sdk.Context):
        config_params = load_config('config/model_predict_config.json')
        self.model_name = config_params['model']
        self.batch_size = config_params['batch_size']
        # 离线评估启动加载模型
        model_path = ctx.get(dubhe_sdk.ContextKey.ENV_MODEL_PATH)
        model_dict = torch.load(model_path)
        self.code_list = model_dict['code_list']
        self.state_dict = model_dict['state_dict']

    @dubhe_sdk.predict()
    def predict(self, ctx: dubhe_sdk.ContextKey):
        self.data_paths = ctx.get(dubhe_sdk.ContextKey.KEY_DATA_PATHS)
        self.debug_dir = ctx.get(dubhe_sdk.ContextKey.KEY_TMP_DIR)
        ctx.log('model predict starting...')
        # prepeare dir
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir, exist_ok=True)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

        prepare_data_args = [
            "--data_paths",
            *self.data_paths,
            "--debug_dir=%s" % self.debug_dir,
            '--predict_method=predict',
        ]
        prepare_data(prepare_data_args)
        df_test = pd.read_csv(os.path.join(self.debug_dir, "test.csv"))
        code2label = {}
        for idx, code in enumerate(self.code_list):
            code2label[code] = idx

        test_dataset = MyDataset(df_test,
                                  code2label,
                                  test_transforms)

        self.output_txt = os.path.join(self.debug_dir, 'pred_result.txt')
        self.output_img = os.path.join(self.debug_dir, 'vis')
        predict_args = [
            '--model_name=%s' % self.model_name,
            '--output_txt=%s' % self.output_txt,
            '--output_img=%s' % self.output_img,
            '--batch_size=%d' % self.batch_size,
            '--progress_init=0.9',
            '--progress_ratio=0.1',
            '--code_list',
            *self.code_list,
            '--model_num=0'
        ]
        ctx.log('model predict args \n %s' % predict_args)
        model_predict(predict_args, test_dataset, self.state_dict, ctx)
        ctx.log('model predict end.')

@dubhe_sdk.pipeline()
class Infer_Model():
    def __init__(self, ctx: dubhe_sdk.Context):
        config_params = load_config('config/model_inference_config.json')
        self.batch_size = config_params['batch_size']
        self.model_name = config_params["Gechuang_aimidware_model_base_or_full_name"]
        # 在线推理启动加载模型
        rely_model_data = ctx.get(dubhe_sdk.ContextKey.ENV_RELY_MODEL_DATA)
        model_info = rely_model_data[self.model_name]
        ckpt_dir = model_info['model_path']
        ctx.log('Model params has been loaded from %s.' % ckpt_dir)
        model_dict = torch.load(ckpt_dir)
        self.code_list = model_dict['code_list']
        self.state_dict = model_dict['state_dict']
        resnet = getattr(mm, config_params["model"])()
        self.model = mm.Net(resnet, config_params["model"],len(self.code_list))
        self.model.load_state_dict(self.state_dict)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

    @dubhe_sdk.inference()
    def inference(self, ctx:dubhe_sdk.ContextKey):
        image_base_dir = ctx.get(dubhe_sdk.ContextKey.KEY_IMAGE_BASE_DIR)
        image_names = ctx.get(dubhe_sdk.ContextKey.KEY_IMAGE_NAMES)
        self.debug_dir = ctx.get(dubhe_sdk.ContextKey.KEY_TMP_DIR)
        ctx.log('model inference starting...')
        # prepeare dir
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir, exist_ok=True)

        full_image_names = [os.path.join(image_base_dir, image_name) for image_name in image_names]
        df_total = pd.DataFrame()
        for img_cache in full_image_names:
            df_total_cache = pd.DataFrame()
            df_total_cache['image'] = [img_cache]
            df_total_cache['code'] = image_base_dir.split('/')[-1]
            df_total = pd.concat([df_total, df_total_cache])
        ctx.log(f'The data set shape is {df_total.shape}.')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
        code2label = {}
        for idx, code in enumerate(self.code_list):
            code2label[code] = idx
        test_dataset = MyDataset(df_total,
                                 code2label,
                                 test_transforms)

        self.output_txt = os.path.join(self.debug_dir, 'pred_result.txt')
        self.output_img = os.path.join(self.debug_dir, 'vis')
        predict_args = [
            '--output_txt=%s' % self.output_txt,
            '--output_img=%s' % self.output_img,
            '--batch_size=%d' % self.batch_size,
            '--progress_init=0.9',
            '--progress_ratio=0.1',
            '--code_list',
            *self.code_list,
            '--model_num=0'
        ]
        ctx.log('model inference args \n %s' % predict_args)
        data = model_inference(predict_args, test_dataset, self.model, ctx)
        ctx.log('model inference end.')
        return data
