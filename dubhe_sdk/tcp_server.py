import json
import socketserver
from concurrent.futures import ThreadPoolExecutor
# from dubhe_sdk.config import IP, PORT
from dubhe_sdk.service.service_prepare import serverIsReady, serviceCheck, connect_tcp
from dubhe_sdk.service.Logger import Logger
from dubhe_sdk.service.service_prepare import *

import struct
import traceback

logger = Logger.instance()

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

class TcpHandler(socketserver.BaseRequestHandler):
    def __init__(self, request, client_address, server):
        self.executor = ThreadPoolExecutor(10)
        super(TcpHandler, self).__init__(request, client_address, server)

    def recvMsg(self):
        head = bytes()
        msgBody = bytes()
        while len(head) < 4:
            head += self.request.recv(4-len(head))
            print(head)
            if (len(head) == 0):
                return

        msgLen = struct.unpack('!i', head)[0]
        while msgLen > 0:
            data = self.request.recv(1024)
            msgBody += data
            msgLen -= len(data)
        return msgBody.decode("utf-8")

    def sendMsg(self, msg):
        bMsg = msg.encode('utf-8')
        length = len(bMsg)
        lenData = struct.pack('!i', length)
        self.request.send(lenData)
        self.request.sendall(bMsg)

    def handle(self):
        logger.info('enter---------------handler')
        while True:
            try:
                data = self.recvMsg()
                if not data:
                    break
                logger.info("recv: %s"%data)
                res = self.dispatcher(data)
                self.sendMsg(res)
                logger.info("send: %s"%res)

            except ConnectionResetError as e:
                logger.error("handle msg error", e)
                break

    def dispatcher(self, msg):
        response = {"flag": 1}
        dict_params = None
        try:
            dict_params = json.loads(msg)
            if(not isinstance(dict_params, dict)):
                response.update({"flag": 2, "message": "param error: request is not a dict"})
                return json.dumps(self.response)

            # instance_id check
            if dict_params['instance_id'] != INSTANCE_ID:
                response.update({"flag": 2, "message": "param error: instance_id [{}] of request different from ENV INSTANCE_ID [{}]!"
                                .format(dict_params['instance_id'],INSTANCE_ID)})
                return json.dumps(self.response)

            reqType = dict_params.get('req_type', -1)
            canOffer, content = serviceCheck(dict_params)
            if canOffer:
                if 0 == reqType:
                    # 心跳检测
                    response["message"] = "service online"
                elif 1 == reqType:
                    # 离线训练
                    self.executor.submit(task_run, dict_params)
                    response["message"] = "model train starting now"
                elif 2 == reqType:
                    # 离线评估
                    self.executor.submit(task_run, dict_params)
                    response["message"] = "model predict starting now"
                elif 3 == reqType:
                    # 在线推理
                    response["message"] = "model inference success"
                    response['request_number'] = dict_params['request_number']
                    response['data'] = task_run(dict_params)
                    if not response['data']:
                        response.update({"flag": 2, "message": "exception occurs"})
                else:
                    response.update({"flag": 2, "message": "req_type error"})
            else:
                response.update({"flag": 2, "message": content})
        except ValueError:
            response.update({"flag": 2, "message": "param error"})

        finally:
            return json.dumps(response, indent=4)
