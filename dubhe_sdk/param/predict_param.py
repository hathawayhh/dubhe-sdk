from dubhe_sdk.param.base_param import BaseParam
from dubhe_sdk.config import *


class PredictParam(BaseParam):
    # 请求类型：1.离线预测 2.离线评估 3.在线推理
    req_type = 2
    # 实例ID
    instance_id = INSTANCE_ID
    # 多个数据集路径
    data_paths = []
    # 模型文件路径
    model_path = model_path
    # 依赖其他模型路径
    rely_model_data = RELY_MODEL_DATA
    # 资源模式：0.CPU模式  1.GPU模式
    resource_type = 1
    # 资源配置信息
    resource_allocation = {
        "CUDA_VISIBLE_DEVICES": "0"  # None, 0, 1, 2, 3
    }
    # 业务参数
    special_params = special_params
    # 中间结果目录
    BASE_DIR = '/temporary/debug'
    # 模型输出目录
    OUTPUT_DIR = 'output'