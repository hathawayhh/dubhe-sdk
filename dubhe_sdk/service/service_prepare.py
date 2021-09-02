# import time
from dubhe_sdk.service.ADCkafka import *
import urllib.request as request
# send pretending msg
import socket
from dubhe_sdk.config import *

from dubhe_sdk.service.Logger import Logger
logger = Logger.instance()


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
        send_kafka(MODEL_READY, ready_json, TOPIC_MODEL_STATUS)
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
    if (PLATFORM_TYPE == AI_PLATFORM):
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
    if (PLATFORM_TYPE == AI_PLATFORM):
        serverIsReady()
    logger.info('server started !')


