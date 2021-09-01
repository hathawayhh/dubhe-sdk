from dubhe_sdk.param.base_param import BaseParam
from dubhe_sdk.config import *


class TrainParam(BaseParam):
    # 请求类型：1.离线预测 2.离线评估 3.在线推理
    req_type = 1
    # 实例ID
    instance_id = INSTANCE_ID
    # 多个数据集路径
    data_paths = []
    # 训练类别
    code_list = []
    # 1：占比   2：指定验证集
    verify_way = 1
    # 验证集占比
    split_rate = 0.1
    # 依赖其他模型路径
    rely_model_data = RELY_MODEL_DATA
    # 资源类型：0.CPU模式  1.GPU模式
    resource_type = 1
    # 资源配置信息
    resource_allocation = {
        "CUDA_VISIBLE_DEVICES": "0"  # None, 0, 1, 2, 3
    }
    # 业务参数
    special_params = special_params
    # 预训练模型路径
    pretrained_model_path = ""
    # 标识断点训练的epoch值，默认0
    pretrained_epoch = 0