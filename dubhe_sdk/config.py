# -*- coding=utf-8 -*-
"""
    Desc:
    Auth: LiZhifeng
    Date: 2020/6/9
"""
import os
import uuid
import json
from dubhe_sdk.pipeline.Logger import ADCLog

isDEBUG = False
logger = ADCLog.getMainLogger()
logger.info('**************config.py*****************')

DEFAULT_TOPIC_MODEL_STATUS = 'alg_model_progress'
DEFAULT_TOPIC_MODEL_RESULT = 'als_model_result'

# 启动配置
if 'PLATFORM_TYPE' in os.environ and int(os.environ['PLATFORM_TYPE']) == 1:
    logger.info('AI Platfrom Started Algorithm!')
else:
    logger.info('Manual Started Algoritm!')

if 'PORT' in os.environ:
    logger.info('PORT %s ' % os.environ['PORT'])
    PORT = int(os.environ['PORT'])
else:
    PORT = 8030
    logger.info('no PORT, use %d' % PORT)
IP = "0.0.0.0"
THREADED = True
if 'INSTANCE_ID' in os.environ:
    logger.info('INSTANCE_ID %s ' % os.environ['INSTANCE_ID'])
    INDEX_URL = "http://%s:%d/heart" % (os.environ['INSTANCE_ID'],PORT)
else:
    INDEX_URL = "http://127.0.0.1:%d/heart" % PORT
    logger.info('no INSTANCE_ID,use {} '.format(INDEX_URL))

# kafka配置
KAFKA_ENABLE = True
if 'BOOTSTRAP_SERVERS' in os.environ:
    logger.info('BOOTSTRAP_SERVERS %s ' % os.environ['BOOTSTRAP_SERVERS'])
    BOOTSTRAP_SERVERS = os.environ['BOOTSTRAP_SERVERS']
else:
    BOOTSTRAP_SERVERS = ['192.168.7.114:9092','192.168.7.113:9092']
    logger.info('no BOOTSTRAP_SERVERS, use %s' % BOOTSTRAP_SERVERS)

# TOPIC配置
topic_name1 = "alg_train_progress"

if 'TOPIC_MODEL_STATUS' in os.environ:
    logger.info('TOPIC_MODEL_STATUS %s' % os.environ['TOPIC_MODEL_STATUS'])
    TOPIC_MODEL_STATUS = os.environ['TOPIC_MODEL_STATUS']
else:
    TOPIC_MODEL_STATUS = DEFAULT_TOPIC_MODEL_STATUS
    logger.info('no TOPIC_MODEL_STATUS, use default %s ' % TOPIC_MODEL_STATUS)

if 'TOPIC_MODEL_RESULT' in os.environ:
    logger.info('TOPIC_MODEL_RESULT  %s ' % os.environ['TOPIC_MODEL_RESULT'])
    TOPIC_MODEL_RESULT = os.environ['TOPIC_MODEL_RESULT']
else:
    TOPIC_MODEL_RESULT = DEFAULT_TOPIC_MODEL_RESULT
    logger.info('no TOPIC_MODEL_RESULT, use default %s ' % TOPIC_MODEL_RESULT)

if 'INSTANCE_ID' in os.environ:
    logger.info('INSTANCE_ID %s' % os.environ['INSTANCE_ID'])
    INSTANCE_ID = os.environ['INSTANCE_ID']
else:
    INSTANCE_ID = str(uuid.uuid1())
    logger.info('no INSTANCE_ID, use %s ' % INSTANCE_ID)

if 'RELY_MODEL_DATA' in os.environ:
    logger.info('RELY_MODEL_DATA %s ' % os.environ['RELY_MODEL_DATA'])
    RELY_MODEL_DATA = json.loads(os.environ['RELY_MODEL_DATA'])
else:
    RELY_MODEL_DATA = {
        "detector":{
            "alg_type":"目标检测",
            "alg_version":"v1.0",
            "model_path":"/temporary/debug/model_20201029082410454_1_0.51.pth",
            "model_version":"v7.0"
        },
        "classify":{
            "alg_type":"分类",
            "alg_version":"v1.0",
            "model_path":"/temporary/debug/model_20201029084953456_4_2.29.pth",
            "model_version":"v7.0"
        }
    }
    logger.info('no RELY_MODEL_DATA, use %s ' % INSTANCE_ID)

if 'SPECIAL_PARAMS' in os.environ:
    logger.info('SPECIAL_PARAMS %s ' % os.environ['SPECIAL_PARAMS'])
    SPECIAL_PARAMS = json.loads(os.environ['SPECIAL_PARAMS'])
else:
    SPECIAL_PARAMS = {
        "section_id": "1E953", #站点ID,
        "debug_model": 0,   #0:关闭, 1:开启
    }

# message type
# status
MES_TRAIN_STATUS = 1
MES_TRAIN_PROGRESS = 2
MES_PREDICT_STATUS = 1
MES_PREDICT_PROGRESS = 2
MES_TRAIN_READY = 0
MES_PREDICT_READY = 0

#result type
MES_RESULT_TRAIN_CLASSIFY = 3
MES_RESULT_TRAIN_DETECTOR = 3
MES_RESULT_INFERENCE_CLASSIFY = 3
MES_RESULT_INFERENCE_DETECTOR = 3

MES_RESULT_COMBINE = 3

# instance id prefix
PREFIX_TRAIN = 'train'
PREFIX_PREDICT = 'forecast'

# task type
TASK_TRAIN_TYPE = 1
TASK_PREDICT_TYPE = 2
