
from kafka import KafkaProducer
import json
from dubhe_sdk.config import *
import uuid
import time
import datetime

def msg_id():
    return uuid.uuid1().int>>64

# 模型状态信息
def model_ready_data():
    """
    {
		"action": "READY" # START, END ,
		"status": 200 # 200: 正常 /300: 异常,
		"msg": "content"
	}
    :return:
    """
    data = {}
    data['action'] = 'READY'
    data['status'] = 200
    data['msg'] = 'model is ready'
    return data

def train_start_data():
    """
    {
		"action": "START" # START, END ,
		"status": 200 # 200: 正常 /300: 异常,
		"msg": "content"
	}
    :return:
    """
    data = {}
    data['action'] = 'START'
    data['status'] = 200
    data['msg'] = 'model train starting'
    return data

def train_end_data():
    """
    {
		"action": "START" # START, END ,
		"status": 200 # 200: 正常 /300: 异常,
		"msg": "content"
	}
    :return:
    """
    data = {}
    data['action'] = 'END'
    data['status'] = 200
    data['msg'] = 'model train end'
    return data

def train_exception_end_data(exceptionMSG):
    """
    {
		"action": "START" # START, END ,
		"status": 200 # 200: 正常 /300: 异常,
		"msg": "content"
	}
    :return:
    """
    data = {}
    data['action'] = 'END'
    data['status'] = 300
    data['msg'] = exceptionMSG
    return data

def predict_start_data():
    """
    {
		"action": "START" # START, END ,
		"status": 200 # 200: 正常 /300: 异常,
		"msg": "content"
	}
    :return:
    """
    data = {}
    data['action'] = 'START'
    data['status'] = 200
    data['msg'] = 'model predict starting'
    return data

def predict_end_data():
    """
    {
		"action": "START" # START, END ,
		"status": 200 # 200: 正常 /300: 异常,
		"msg": "content"
	}
    :return:
    """
    data = {}
    data['action'] = 'END'
    data['status'] = 200
    data['msg'] = 'model predict end'
    return data

def predict_exception_end_data(exceptionMSG):
    """
    {
		"action": "START" # START, END ,
		"status": 200 # 200: 正常 /300: 异常,
		"msg": "content"
	}
    :return:
    """
    data = {}
    data['action'] = 'END'
    data['status'] = 300
    data['msg'] = exceptionMSG
    return data

# 模型进度信息
def train_progress_data(progress, epoch, metrics=dict()):
    """
    {
        "progress": 0.2 # 0-1 float ,
        "epoch": 0 # 0 - total epochs,
        "metrics": {
                "loss": 1.1 # float,
                "lr": 0.001 # float ,
                "iou": # float,
                "acc": 0.5 # 0-1 float,
                "recall": 0.5 # 0-1 float
            } # float,
	}
    :param progress:
    :param epoch:
    :param metrics:
    :return:
    """

    data = metrics
    data['progress'] = progress
    data['epoch'] = epoch
    return data

def predict_progress_data(progress):
    """
    {
        "progress": 0.2 # 0-1 float ,
	}
    :param progress:
    :return:
    """
    data = {}
    data['progress'] = progress
    return data

def predict_classify_details(img_path, ground_true, predict_code, conf):
    """
    {
		"img_path": "img_full_path"  ,
		"ground_true": "code1" # ,
		"predict_code": "code2",
		"conf": 0.8 # 0-1 float ,
	}
    :return:
    """
    data_list = []
    data = {}
    data['ground_true'] = ground_true
    data['predict_code'] = predict_code
    data['conf'] = conf
    data_list.append(data)

    result = {}
    result['img_path'] = img_path
    result['info'] = data_list
    return [result]

def predict_details(img_path, ground_true, predict_code, conf, shape_type=None,position=None,special=None):
    """
	{
		"img_path": "img_full_path"  ,
		"pre_img": "preimg_full_path"  ,
		"ground_true": "code1" # ,
		"predict_code": "code2",
		"conf": 0.8 # 0-1 float ,
	}
    :return:
    """
    data_list = []
    if not position:
        data = {}
        data['ground_true'] = ground_true
        data['predict_code'] = predict_code
        data['conf'] = conf
    elif len(position) > 0:
        for pos in position:
            (x1, y1, x2, y2, boxconf) = pos
            data = {}
            data['ground_true'] = ground_true
            data['predict_code'] = predict_code
            data['conf'] = conf
            if conf<0:
                data['conf'] = boxconf
            data['shape_type'] = shape_type
            data['position'] = [[x1,y1], [x2,y2]]
            data['special'] = special
            data_list.append(data)
    else:
        data = {}
        data['ground_true'] = ground_true
        data['predict_code'] = predict_code
        data['conf'] = conf
        if conf < 0:
            data['conf'] = 1
        data['shape_type'] = shape_type
        data['position'] = []
        data['special'] = special

    data_list.append(data)
    result = {}
    result['img_path'] = img_path
    if position:
        result['bbox_cnt'] = len(position)
    result['info'] = data_list
    return [result]


def msg_context(mes_type, data):
    kafka_json = {}
    kafka_json['instance_id'] = KafkaParam.instance_id
    kafka_json['request_number'] = msg_id()
    kafka_json['date'] = int(time.time()*1000)
    kafka_json['mes_type'] = mes_type
    kafka_json['data'] = data
    return kafka_json

def send_kafka(mes_type, data, topic=topic_name1, local_backup='json.dat'):
    """
    :param topic:
    :param message: json format
    :return:
    """
    msg = msg_context(mes_type, data)
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    with open(local_backup, encoding='utf-8', mode='a+') as writer:
        writer.write('%s, %s, %s\n' % (time_str, topic, msg))
    if KAFKA_ENABLE:
        try:
            producer = KafkaProducer(bootstrap_servers=BOOTSTRAP_SERVERS,
                                     value_serializer=lambda m: json.dumps(m).encode('utf-8'))
            producer.send(topic, value=msg)
            producer.close()
        except Exception as e:
            print(msg)
            print(e)
            raise e

class KafkaParam(object):
    instance_id = INSTANCE_ID

