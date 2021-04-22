from dubhe_sdk.http_server import app, Model, set_model
# import dubhe_sdk.http_server as server
from dubhe_sdk.config import *
from dubhe_sdk.utils import *
from dubhe_sdk.pipeline.ADCkafka import *
from dubhe_sdk.param.train_param import TrainParam
from dubhe_sdk.utils import load_config

from geek_vision_algo.train import main as model_train
from geek_vision_algo.predict import main as model_predict

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split
import glob
import re
import shutil
import traceback

# model对象，重写 train，predict方法
class My_Model(Model):
    def __init__(self):
        super().__init__()

    def exception_end(self, msg):
        end_json = train_exception_end_data(msg)
        send_kafka(MES_TRAIN_STATUS, end_json, TOPIC_MODEL_STATUS, os.path.join(self.resultDir, 'json.dat'))

    def train_run(self):
        end_json = train_end_data()
        send_kafka(MES_TRAIN_STATUS, end_json, TOPIC_MODEL_STATUS, os.path.join(self.resultDir, 'json.dat'))

    def train(self, input_params):
        # 入参与config参数结合
        try:
            params = TrainParam._default_values()
            config_params = load_config('config/model_train_config.json')
            self.config_dict_replace(params, config_params)
            self.input_dict_replace(params, input_params)
            logger.info('_params\n')
            logger.info(self.params)

            self.train_params = TrainParam()
            self.train_params.__dict__ = params

            # prepeare dir
            for tmpDir in self.localDirs:
                if not os.path.exists(tmpDir):
                    os.makedirs(tmpDir)
            if not os.path.exists(self.outputModelDir):
                os.makedirs(self.outputModelDir)

             # send start to kafka
            start_json = train_start_data()
            send_kafka(MES_TRAIN_STATUS, start_json, TOPIC_MODEL_STATUS, os.path.join(self.resultDir, 'json.dat'))
            self.logfile = os.path.join(self.logsDir, 'log.txt')
            self.logger = ADCLog.GetLogger(self.logfile)

            # split train and test
            self.logger.info('split train and test starting...')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            whole_set = datasets.ImageFolder(
                os.path.join(self.train_params.data_paths[0], "Images"),
                transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(p=0),
                    transforms.ColorJitter(brightness=0, contrast=0, saturation=0,
                                           hue=0),
                    transforms.ToTensor(),
                    normalize,
                ]))

            length = len(whole_set)
            test_size = int(self.train_params.split_rate * length)
            train_size = length - test_size
            train_set, test_set = random_split(whole_set, [train_size, test_size])

            self.train_set = train_set
            self.test_set = test_set
            self.logger.info('split train and test end.')

            # set gpu id
            os.environ['CUDA_VISIBLE_DEVICES'] = self.train_params.resource_allocation['CUDA_VISIBLE_DEVICES']
            self.logger.info('CUDA_VISIBLE_DEVICES = %s' % os.environ['CUDA_VISIBLE_DEVICES'])

            # train model
            self.logger.info('model train starting...')
            train_args = [
                '--init_epoch=%d' % self.train_params.pretrained_epoch,
                '--epochs=%d' % self.train_params.epochs,
                '--batch_size=%d' % self.train_params.batch_size,
                '--lr=%f' % self.train_params.lr,
                '--evaluation_interval=%d' % self.train_params.evaluation_interval,
                '--save_path=%s' % self.modelFileDir,
                '--progress_init=0.0',
                '--progress_ratio=0.9',
                '--model_optimizer=SGD',
                '--class_num=%d' % len(self.train_params.code_list),
                '--model_num=0'
            ]
            self.logger.info('model train args \n %s ' % train_args)
            model_train(train_args, self.train_set, self.test_set)  # please design your model train logic in this function.
            self.logger.info('model train end.')

            # 2. find best model file
            model_files = glob.glob(os.path.join(self.modelFileDir, '*'))
            model_prefix = r'model_(\d)+_{}'.format(self.train_params.epochs - 1)
            for file in model_files:
                model_name = os.path.basename(file)
                if re.match(model_prefix, model_name, re.M | re.I):

                    self.model_file = model_name
                    break

            # 3. predict model
            self.logger.info('model predict starting...')
            self.output_txt = os.path.join(self.resultDir, 'pred_result.txt')
            self.output_img = os.path.join(self.resultDir, 'vis')
            predict_args = [
                '--pretrained_weights=%s' % os.path.join(self.modelFileDir, self.model_file),
                '--output_txt=%s' % self.output_txt,
                '--output_img=%s' % self.output_img,
                '--predict_method=train',
                '--progress_init=0.9',
                '--progress_ratio=0.1',
                '--code_list',
                *self.train_params.code_list,
                '--model_num=0'
            ]
            self.logger.info('model predict args \n %s' % predict_args)
            model_predict(predict_args, self.test_set)
            self.logger.info('model predict end.')

            # 3. copy to output modeldir
            files = glob.glob(os.path.join(self.modelFileDir, '*'))
            for _file in files:
                if os.path.isfile(_file):
                    shutil.copy(_file, self.outputModelDir)

            self.train_run()

        except Exception as ex:
            traceback.print_exc()
            exceptionMSG = traceback.format_exc()
            self.exception_end('%s train model error %s' % (INSTANCE_ID, ex))
            logger.error(exceptionMSG)

    def predict(input_params):
        pass

    def inference(input_params):
        pass

if __name__ == '__main__':
    my_model = My_Model()
    set_model('name',my_model)

    app.run(host=IP, port=PORT, threaded=THREADED)