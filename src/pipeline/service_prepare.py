
# import time
from pipeline.ADCkafka import *
import urllib.request as request
# send pretending msg
from pipeline.Logger import ADCLog
import socket
from config import *


logger = ADCLog.getMainLogger()

def taskType():
    curType = INSTANCE_ID.split('-')[1]
    if curType == PREFIX_TRAIN:
        return TASK_TRAIN_TYPE
    elif curType == PREFIX_PREDICT:
        return TASK_PREDICT_TYPE
    else:
        return -1


def serviceCheck(dict_params):

    # 1:gpu , 0:cpu
    resource_type = dict_params['resource_type']
    # req_type = 0: heart, 1: train,2: offline inference, 3: online inference
    req_type = dict_params['req_type']
    # train cpu is not support now req_type=1
    if req_type == 1 and resource_type == 0:
        return False, 'train cpu is not support now'

    return True, 'sevice online'


def serverIsReady():
    print("enter serverIsReady")
    if KAFKA_ENABLE:
        print("prepara ready flag.")
        ready_json = model_ready_data()
        send_kafka(MES_PREDICT_READY, ready_json, TOPIC_MODEL_STATUS)
        print("ready send success!")



def open_browser():
    logger.info('server starting...')
    while True:
        try:
            request.urlopen(url=INDEX_URL)
            break
        except Exception as e:
            print(e)
            time.sleep(0.5)
    serverIsReady()
    logger.info('server started !')

def connect_tcp(host, port):
    logger.info('server starting...')
    while True:
        # logger.info("****")
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            code = client.connect_ex((host,port))
            client.close()
            if code == 0:
                break
        except Exception as e:
            print(e)
            time.sleep(1)
    serverIsReady()
    logger.info('server started !')


