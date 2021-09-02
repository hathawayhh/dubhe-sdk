# -*- coding=utf-8 -*-
"""
    Desc:
    Auth: LiZhifeng
    Date: 2020/6/9
"""
import os
import uuid
import json
from dubhe_sdk.service.Logger import Logger

logger = Logger.instance()

isDEBUG = False

logger.info('**************config.py*****************')

DEFAULT_TOPIC_MODEL_STATUS = 'alg_model_progress'
DEFAULT_TOPIC_MODEL_RESULT = 'als_model_result'
IP = "0.0.0.0"
# kafka配置
KAFKA_ENABLE = True
THREADED = True
# TOPIC配置
topic_name1 = "alg_train_progress"

# 启动配置
DEBUG_PLATFORM = 0
AI_PLATFORM = 1

if 'PLATFORM_TYPE' in os.environ and int(os.environ['PLATFORM_TYPE']) == AI_PLATFORM:
    logger.info('AI Platfrom Started Algorithm!')
    logger.info('---------------ENV------------------')
    PLATFORM_TYPE = AI_PLATFORM

    PORT = int(os.environ['PORT'])
    logger.info('PORT %s ' % os.environ['PORT'])

    BOOTSTRAP_SERVERS = os.environ['BOOTSTRAP_SERVERS']
    logger.info('BOOTSTRAP_SERVERS %s ' % BOOTSTRAP_SERVERS)

    TOPIC_MODEL_STATUS = os.environ['TOPIC_MODEL_STATUS']
    logger.info('TOPIC_MODEL_STATUS %s' % TOPIC_MODEL_STATUS)

    TOPIC_MODEL_RESULT = os.environ['TOPIC_MODEL_RESULT']
    logger.info('TOPIC_MODEL_RESULT  %s ' % TOPIC_MODEL_RESULT)

    INSTANCE_ID = os.environ['INSTANCE_ID']
    logger.info('INSTANCE_ID %s' % INSTANCE_ID)

    INDEX_URL = "http://%s:%d/heart" % (os.environ['INSTANCE_ID'], PORT)
    logger.info('INDEX_URL %s ' % INDEX_URL)

    RELY_MODEL_DATA = None
    if 'RELY_MODEL_DATA' in os.environ and os.environ['RELY_MODEL_DATA'] != None and len(os.environ['RELY_MODEL_DATA']) != 0:
        RELY_MODEL_DATA = json.loads(os.environ['RELY_MODEL_DATA'])
        logger.info('RELY_MODEL_DATA %s ' % RELY_MODEL_DATA)

    special_params = None
    if 'special_params' in os.environ and os.environ['special_params'] != None and len(os.environ['special_params']) != 0:
        special_params = json.loads(os.environ['special_params'])
        logger.info('special_params %s ' % special_params)

    model_path = None
    if 'model_path' in os.environ and os.environ['model_path'] != None:
        model_path = os.environ['model_path']
        logger.info('model_path %s ' % model_path)


    logger.info('---------------ENV------------------')
else:
    logger.info('Manual Started Algoritm!')
    logger.info('---------------DEFAULT------------------')
    PLATFORM_TYPE = DEBUG_PLATFORM
    IP = "127.0.0.1"
    PORT = 8030
    logger.info('Default IP: %s, PORT: %d' %(IP,PORT))

    # BOOTSTRAP_SERVERS = ['192.168.7.114:9092', '192.168.7.113:9092']
    BOOTSTRAP_SERVERS = '127.0.0.1:9092'
    logger.info('Default BOOTSTRAP_SERVERS, use %s' % BOOTSTRAP_SERVERS)

    TOPIC_MODEL_STATUS = DEFAULT_TOPIC_MODEL_STATUS
    logger.info('Default TOPIC_MODEL_STATUS, use default %s ' % TOPIC_MODEL_STATUS)

    TOPIC_MODEL_RESULT = DEFAULT_TOPIC_MODEL_RESULT
    logger.info('Default TOPIC_MODEL_RESULT, use default %s ' % TOPIC_MODEL_RESULT)

    INSTANCE_ID = "modelid1412-algorithm112345540-train-1412-4"
    # INSTANCE_ID = "forecastid323-algorithm154146065-forecast-323-4"
    # INSTANCE_ID = "taskid14-application203913747-inference-14"
    logger.info('Default INSTANCE_ID, use %s ' % INSTANCE_ID)

    INDEX_URL = "http://127.0.0.1:%d/heart" % PORT
    logger.info('Default INDEX_URL, use {} '.format(INDEX_URL))

    # RELY_MODEL_DATA = {
    #     "detector": {
    #         "alg_type": "目标检测",
    #         "alg_version": "v1.0",
    #         "model_path": "/temporary/debug/model_20201029082410454_1_0.51.pth",
    #         "model_version": "v7.0"
    #     },
    #     "classify": {
    #         "alg_type": "分类",
    #         "alg_version": "v1.0",
    #         "model_path": "/temporary/debug/model_20201029084953456_4_2.29.pth",
    #         "model_version": "v7.0"
    #     }
    # }

    # RELY_MODEL_DATA = {
    #     "unet": {
    #         "alg_type": "目标检测",
    #         "alg_version": "v1.0",
    #         "model_path": "./ckpts/Unet_se_resnext50_32x4d_lowest_loss.pth",
    #         "model_version": "v7.0"
    #     }
    # }

    RELY_MODEL_DATA = {
        "classification_model": {
            "alg_type": "目标检测",
            "alg_version": "v1.0",
            "model_path": "./output/models/model_20210830140750571_1_0.00.pth",
            "model_version": "v7.0"
        }
    }

    logger.info('Default RELY_MODEL_DATA, use %s ' % RELY_MODEL_DATA)

    special_params = {
        "section_id": "1E953",  # 站点ID,
        "debug_model": 0,  # 0:关闭, 1:开启
    }
    # model_path = None
    model_path = "./output/models/model_20210830140750571_1_0.00.pth"


    logger.info('---------------DEFAULT------------------')


# message type
# status
MODEL_READY = 0
MODEL_STATUS = 1
MODEL_PROGRESS = 2

#result type
MODEL_RESULT = 3

# instance id prefix
PREFIX_TRAIN = 'train'
PREFIX_PREDICT = 'forecast'
PREFIX_INFERENCE = 'inference'
PREFIX_AUTOMARK = 'automark'

# task type
TASK_TRAIN_TYPE = 1
TASK_PREDICT_TYPE = 2
TASK_INFERENCE_TYPE = 3

if INSTANCE_ID.split('-')[2] == PREFIX_TRAIN:
    TASK_TYPE = TASK_TRAIN_TYPE
elif INSTANCE_ID.split('-')[2] == PREFIX_PREDICT:
    TASK_TYPE =  TASK_PREDICT_TYPE
elif INSTANCE_ID.split('-')[2] == PREFIX_INFERENCE or INSTANCE_ID.split('-')[2] == PREFIX_AUTOMARK:
    TASK_TYPE =  TASK_INFERENCE_TYPE
else:
    logger.error("TASK %s PREFIX not defined!"%INSTANCE_ID.split('-')[2])
    TASK_TYPE =  -1
