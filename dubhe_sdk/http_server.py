import os
import sys

from flask import Flask, jsonify, request as flask_request
from concurrent.futures import ThreadPoolExecutor
from dubhe_sdk.config import *
import json
from dubhe_sdk.service.Logger import Logger
logger = Logger.instance()
from dubhe_sdk.service.service_prepare import *
import traceback

executor = ThreadPoolExecutor(10)
app = Flask(__name__)

model = {}

def getModel(name="entrance_func", defValue=None):
    try:
        return model[name]
    except KeyError:
        return defValue

def setModel(name="entrance_func", value=None):
    model[name] = value

def task_run(dict_params):

    try:
        data = None
        service_type, func, ctxb = getModel()
        assert service_type == dict_params['req_type'], "pod service_type are different from req_type!"

        # send start to kafka
        if dict_params['req_type'] != TASK_INFERENCE_TYPE and PLATFORM_TYPE == AI_PLATFORM:
            start_json = train_start_data()
            send_kafka(MODEL_STATUS, start_json, TOPIC_MODEL_STATUS)

        # set gpu id
        resource_allocation = dict_params['resource_allocation']
        os.environ['CUDA_VISIBLE_DEVICES'] = resource_allocation['CUDA_VISIBLE_DEVICES']

        ctx = ctxb.setInputParam(dict_params).build()
        data = func(ctx)

        # send end to kafka
        if dict_params['req_type'] != TASK_INFERENCE_TYPE and PLATFORM_TYPE == AI_PLATFORM:
            end_json = train_end_data()
            send_kafka(MODEL_STATUS, end_json, TOPIC_MODEL_STATUS)

    except Exception as ex:
        traceback.print_exc()
        exceptionMSG = traceback.format_exc()
        if dict_params['req_type'] != TASK_INFERENCE_TYPE and PLATFORM_TYPE == AI_PLATFORM:
            end_json = train_exception_end_data('%s current model error %s' % (INSTANCE_ID, ex))
            send_kafka(MODEL_STATUS, end_json, TOPIC_MODEL_STATUS)
        logger.error(exceptionMSG)
    finally:
        return data

# region 0. heart detection
@app.route('/heart', methods=['GET', 'POST'])
def heart():
    res = {
            "flag":1,
            "message": "service online"
            }
    return jsonify(res)
# endregion

# region 1. offline train
@app.route('/train/start', methods=['GET', 'POST'])
def train():
    params = flask_request.get_json()
    logger.info("input params:{}".format(params))
    if type(params) == type(dict()):
        dict_params = params
    else:
        dict_params = json.loads(params)
    print('/train/start %s\n' % dict_params)
    # instance_id check
    if dict_params['instance_id'] != INSTANCE_ID:
        res = {
            "flag": 2,
            "message": "param error: instance_id [{}] of request different from ENV INSTANCE_ID [{}]!"
            .format(dict_params['instance_id'], INSTANCE_ID)
        }
        return jsonify(res)

    executor.submit(task_run, dict_params)
    res = {
            "flag":1,
            "message": "model train starting now"
            }
    return jsonify(res)
# endregion

# region 2. offline evaluate
@app.route('/predict/batch', methods=['GET', 'POST'])
def predict_batch():
    params = flask_request.get_json()
    logger.info("input params:{}".format(params))
    if type(params) == type(dict()):
        dict_params = params
    else:
        dict_params = json.loads(params)
    print('/predict/batch %s\n' % dict_params)

    # instance_id check
    if dict_params['instance_id'] != INSTANCE_ID:
        res = {
            "flag": 2,
            "message": "param error: instance_id [{}] of request different from ENV INSTANCE_ID [{}]!"
                .format(dict_params['instance_id'], INSTANCE_ID)
        }
        return jsonify(res)

    executor.submit(task_run, dict_params)
    res = {
            "flag":1,
            "message": "model predict starting now"
            }
    return jsonify(res)
# endregion

# region 3. online inference
@app.route('/predict/multiple', methods=['GET', 'POST'])
def predict_multiple():
    params = flask_request.get_json()
    logger.info("input params:{}".format(params))
    if type(params) == type(dict()):
        dict_params = params
    else:
        dict_params = json.loads(params)
    print('/predict/multiple %s\n' % dict_params)

    # instance_id check
    if dict_params['instance_id'] != INSTANCE_ID:
        res = {
            "flag": 2,
            "message": "param error: instance_id [{}] of request different from ENV INSTANCE_ID [{}]!"
                .format(dict_params['instance_id'], INSTANCE_ID)
        }
        return jsonify(res)

    data = task_run(dict_params)
    res = {
        "flag": 1,
        "message": "model inference success",
        "data": data
    }

    logger.info("response:{}".format(json.dumps(res, indent=4)))
    return jsonify(res)
# endregion


if __name__ == '__main__':
    app.run(host=IP, port=PORT, threaded=THREADED)
    # send ready to kafka
