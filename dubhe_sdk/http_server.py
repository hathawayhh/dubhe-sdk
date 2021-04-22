import os
import sys

# curPath = os.path.abspath(os.path.dirname(__file__))
# print(curPath)
# sys.path.insert(0, curPath)
from flask import Flask, jsonify, request as flask_request
from concurrent.futures import ThreadPoolExecutor
from dubhe_sdk.config import *
from dubhe_sdk.pipeline.service import Model
import json
from dubhe_sdk.pipeline.Logger import ADCLog
logger = ADCLog.getMainLogger()

executor = ThreadPoolExecutor(10)
app = Flask(__name__)
from dubhe_sdk.pipeline.service_prepare import *
from dubhe_sdk.pipeline.service_prepare import open_browser
executor.submit(open_browser)
from dubhe_sdk.param.train_param import TrainParam
from dubhe_sdk.param.predict_param import PredictParam
from dubhe_sdk.param.inference_param import InferenceParam


model = {}

def set_model(name, value):
    model[name] = value

def get_model(name, defValue=None):
    try:
        return model[name]
    except KeyError:
        return defValue

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

    get_model('name').train(dict_params)
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

    executor.submit(model.predict, dict_params)
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
    res = model.inference(dict_params)
    logger.info("response:{}".format(res))
    return jsonify(res)
# endregion


if __name__ == '__main__':
    app.run(host=IP, port=PORT, threaded=THREADED)
    # send ready to kafka
