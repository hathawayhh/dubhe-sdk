from dubhe_sdk.param.train_param import TrainParam
from dubhe_sdk.param.predict_param import PredictParam
from dubhe_sdk.param.inference_param import InferenceParam
from dubhe_sdk.service.ADCkafka import *
from dubhe_sdk.config import *
import typing
import logging


class ContextKey:
    """
    Key to get algorithm id from context, Str.
    """
    KEY_ALGORITHM_ID = "algorithm_id"

    """
    Key to get data paths from context, List.
    If can not to start with this epoch ( for example, no such saved model file in cache folder ), raise an exception
    """
    KEY_DATA_PATHS = "data_paths"

    """
    Key to get train code list from context, List.
    """
    KEY_CODE_LIST = "code_list"
    KEY_VERIFY_WAY = "verify_way"
    KEY_VERIFY_PATHS = "verify_paths"
    KEY_SPLIT_RATE = "split_rate"
    KEY_PRETRAINED_MODEL_PATH = "pretrained_model_path"
    KEY_PRETRAINED_EPOCH = "pretrained_epoch"
    KEY_TMP_DIR = "tmp_dir"
    KEY_OUTPUT_DIR = "output_dir"
    KEY_IMAGE_BASE_DIR = "image_base_dir"
    KEY_IMAGE_NAMES = "image_names"
    KEY_RESOURCE_ALLOCATION = "resource_allocation"
    ENV_INSTANCE_ID = "INSTANCE_ID"
    ENV_PLATFORM_TYPE = "PLATFORM_TYPE"
    ENV_RELY_MODEL_DATA = "RELY_MODEL_DATA"
    ENV_MODEL_PATH = "model_path"
    ENV_SPECIAL_PARAMS = "special_params"

    class ContextKeyError(TypeError):
        pass

    def __setattr__(self, key, value):
        if key in self.__dict__:
            raise self.ContextKeyError(f'Can not set ContextKey {key}')
        self.__dict__[key] = value


class Context(object):
    def __init__(self, logger):
        self.logger = logger
        self.local_logger = logging.getLogger("dubhe")
        self.local_logger.setLevel(logging.INFO)

        self.key_map = {}
        self.ds_map = {}
        self.out_dataset = None

    def get(self, key: str, default: str = ""):
        if key in self.key_map:
            return self.key_map[key]
        else:
            return default

    def log(self, msg, *args, **kwargs):
        self.local_logger.info(msg, *args, **kwargs)

        if self.logger is not None:
            pass

    def logProgress(self, progress: float):

        logstr = "Progress:{:.2%}, ".format(progress)
        self.local_logger.info(logstr)
        if PLATFORM_TYPE == AI_PLATFORM:
            progress_json = predict_progress_data(progress)
            send_kafka(MODEL_PROGRESS, progress_json, TOPIC_MODEL_STATUS)

        if self.logger is not None:
            # todo, send log to cloud
            pass

    def logProgressByBatch(self, epoch: int, batch_idx, batchsize, progress: float, Metrics: typing.Dict[str, typing.Union[float, int]]):

        logstr = "Epoch:{:d} [{:d}|{:d}], Progress:{:.2%}, ".format(epoch, batch_idx, batchsize, progress)
        for m in Metrics.items():
            logstr += "%s: %4s, "%(m[0],m[1])
        self.local_logger.info(logstr)
        if PLATFORM_TYPE == AI_PLATFORM:
            progress_json = train_progress_data(progress, epoch, Metrics)
            send_kafka(MODEL_PROGRESS, progress_json, TOPIC_MODEL_STATUS)

        if self.logger is not None:
            # todo, send log to cloud
            pass

    def logProgressByEpoch(self, epoch: int, progress: float, Metrics: typing.Dict[str, typing.Union[float, int]]):
        logstr = "Epoch:{:d}, Progress:{:.2%}, ".format(epoch, progress)
        for m in Metrics.items():
            logstr += "%s: %4s, "%(m[0],m[1])
        self.local_logger.info(logstr)

        if PLATFORM_TYPE == AI_PLATFORM:
            progress_json = train_progress_data(progress, epoch, Metrics)
            send_kafka(MODEL_PROGRESS, progress_json, TOPIC_MODEL_STATUS)

    def logResult(self, predict_json: typing.Dict):
        logstr = "Result data:{}, ".format(predict_json)
        self.local_logger.info(logstr)

        if PLATFORM_TYPE == AI_PLATFORM and TASK_TYPE != TASK_INFERENCE_TYPE:
            send_kafka(MODEL_RESULT, predict_json, TOPIC_MODEL_RESULT)

    def logResultDetails(self, img_path, ground_true, predict_code, conf, shape_type=None, position=None, special=None):
        predict_json = predict_details(img_path, ground_true, predict_code, conf, shape_type, position, special)
        logstr = "Result data:{}, ".format(predict_json)
        self.local_logger.info(logstr)

        if PLATFORM_TYPE == AI_PLATFORM and TASK_TYPE != TASK_INFERENCE_TYPE:
            send_kafka(MODEL_RESULT, predict_json, TOPIC_MODEL_RESULT)
        return predict_json



class ContextBuilder(object):
    def __init__(self, logger):
        self.ctx = Context(logger)

    def setInputParam(self, dict_params: dict):
        # 模型输出目录
        OUTPUT_MODEL_DIR = 'output/models'
        if dict_params['req_type'] == TASK_TRAIN_TYPE:
            inputParams = TrainParam()
            inputParams.__dict__ = dict_params
            self.ctx.key_map[ContextKey.KEY_DATA_PATHS] = inputParams.data_paths
            self.ctx.key_map[ContextKey.KEY_CODE_LIST] = inputParams.code_list
            self.ctx.key_map[ContextKey.KEY_SPLIT_RATE] = inputParams.split_rate
            self.ctx.key_map[ContextKey.KEY_OUTPUT_DIR] = OUTPUT_MODEL_DIR
            self.ctx.key_map[ContextKey.KEY_PRETRAINED_MODEL_PATH] = inputParams.pretrained_model_path
            self.ctx.key_map[ContextKey.KEY_PRETRAINED_EPOCH] = inputParams.pretrained_epoch

        elif dict_params['req_type'] == TASK_PREDICT_TYPE:
            inputParams = PredictParam()
            inputParams.__dict__ = dict_params
            self.ctx.key_map[ContextKey.KEY_DATA_PATHS] = inputParams.data_paths

        elif dict_params['req_type'] == TASK_INFERENCE_TYPE:
            inputParams = InferenceParam()
            inputParams.__dict__ = dict_params
            self.ctx.key_map[ContextKey.KEY_IMAGE_BASE_DIR] = inputParams.image_base_dir
            self.ctx.key_map[ContextKey.KEY_IMAGE_NAMES] = inputParams.image_names
            self.ctx.key_map[ContextKey.KEY_RESOURCE_ALLOCATION] = inputParams.resource_allocation

        return self

    def add(self, key: str, value):
        self.ctx.key_map[key] = value
        return self

    def build(self):
        return self.ctx





