# -*- coding: UTF-8 -*-

import sys
import os
import functools

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
print(rootPath)
sys.path.insert(0, rootPath)
# logger.info(os.environ)
from pipeline.Logger import ADCLog
from config import *
from pipeline.ADCkafka import *
import traceback

logger = ADCLog.getMainLogger()

def dict_replace(dist, data):
    for key, value in data.items():

        if key in dist and (type(value) == type(dict())):
            dict_replace(dist[key], value)
        else:
            dist[key] = value

def load_config(config_file):
    try:
        with open(config_file) as file:
            config = json.load(file)
            return config
    except OSError as e:
        logger.info(e)
        return None


class Model:
    def __init__(self):
        self.name = 'model'
        self.instance_id = INSTANCE_ID
        self.TEMP_DIR = '/temporary/debug' # 中间结果目录
        self.OUTPUT_DIR = 'output' # 模型输出
        self.localDirs = [
            os.path.join(self.TEMP_DIR, '%s_%s' % (self.name, self.instance_id), 'dataset'),
            os.path.join(self.TEMP_DIR, '%s_%s' % (self.name, self.instance_id), 'models'),
            os.path.join(self.TEMP_DIR, '%s_%s' % (self.name, self.instance_id), 'results'),
            os.path.join(self.TEMP_DIR, '%s_%s' % (self.name, self.instance_id), 'logs')]

        self.datasetDir = self.localDirs[0]
        self.modelFileDir = self.localDirs[1]
        self.resultDir = self.localDirs[2]
        self.logsDir = self.localDirs[3]
        self.outputModelDir = os.path.join(self.OUTPUT_DIR, 'models')

    def config_dict_replace(self, params, config_params):
        """replace dict config_params to dict params"""
        self.config = config_params
        for key, value in self.config.items():
            if key in params and (type(value) == type(dict())):
                dict_replace(params[key], value)
            else:
                params[key] = value
        self.params = params

    def input_dict_replace(self, params, input_params):
        """replace dict input params to dict params"""
        self.input = input_params
        for key, value in input_params.items():
            if key in params and (type(value) == type(dict())):
                dict_replace(params[key], value)
            else:
                params[key] = value
        self.params = params

    def train(self, input_params):
        """Construct the train. """
        raise NotImplementedError('Must be implemented by the subclass.')

    def predict(input_params):
        """Construct the predict. """
        raise NotImplementedError('Must be implemented by the subclass.')

    def inference(input_params):
        """Construct the inference. """
        raise NotImplementedError('Must be implemented by the subclass.')



def main(argv):
    #train({
    #"req_type":1,
    #"data_paths": [
        #"/data/train_data"
    #],
    #"code_list": [],
    #"split_rate": 0.6,
    #"resource_type": 1,
    #"resource_allocation":{
        #"CUDA_VISIBLE_DEVICES": "0"
    #}
    #})
    
    #predict({
    #"req_type":2,
    #"data_paths": [
        #"/data/train_data"
    #],
    #"model_path": "/temporary/debug/model_20201029082410454_1_0.51.pth",
    #"resource_type": 1,
    #"resource_allocation":{
        #"CUDA_VISIBLE_DEVICES": "0"
    #}
    #})
    
    inference({
    "req_type":3,
    "data_paths": [
        "/data/train_data"
    ],
    "model_path": "output/model_20201015093746577_0_6.66.pth",
    "resource_type": 1,
    "resource_allocation":{
        "CUDA_VISIBLE_DEVICES": "0"
    }
    })


if __name__ == "__main__":
    logger.info(sys.argv)
    main(sys.argv[1:])
