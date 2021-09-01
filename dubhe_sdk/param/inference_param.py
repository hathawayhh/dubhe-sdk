from dubhe_sdk.param.base_param import BaseParam
from dubhe_sdk.config import *


class InferenceParam(BaseParam):
    # 请求类型：1.离线预测 2.离线评估 3.在线推理
    req_type = 3
    # 实例ID
    instance_id = INSTANCE_ID
    # 图片路径
    image_base_dir = ''
    # 图片名称
    image_names = []
    # 模型文件路径
    model_path = model_path
    # 唯一的请求ID
    request_number = 0
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