from dubhe_sdk.config import *
from dubhe_sdk.service.ADCkafka import *
from dubhe_sdk.service import context
from dubhe_sdk.service.service_prepare import *
from concurrent.futures import ThreadPoolExecutor
import socketserver

import sys
import importlib
import random
import typing
import inspect
import traceback


def image_run():
    try:
        serve_flag = False
        main_file_path = sys.argv[0]
        pipelines = _find_pipelines(main_file_path)
        entrances = _find_entrances(pipelines)

        # 中间结果目录
        TMP_DIR = '/temporary/debug'

        ctxb = context.ContextBuilder(None)\
            .add(context.ContextKey.ENV_INSTANCE_ID, INSTANCE_ID)\
            .add(context.ContextKey.ENV_PLATFORM_TYPE, PLATFORM_TYPE)\
            .add(context.ContextKey.ENV_RELY_MODEL_DATA, RELY_MODEL_DATA)\
            .add(context.ContextKey.ENV_MODEL_PATH, model_path)\
            .add(context.ContextKey.ENV_SPECIAL_PARAMS, special_params)\
            .add(context.ContextKey.KEY_TMP_DIR, TMP_DIR)
        ctx = ctxb.build()

        for entrance in entrances:
            func_type = entrance[2]
            # 训练：modelid1412-algorithm112345540-train-1412-4
            # 推理：taskid84-application170644901-inference-84
            if func_type == TASK_TYPE:
                module = entrance[0]
                name = entrance[1]
                user_Model = module.__getattribute__(name)
                Model = user_Model(ctx)
                function = entrance[3]
                user_func = function.__get__(Model)
                model_infos = (func_type, user_func, ctxb)

                if 'protocol' in os.environ and int(os.environ['protocol']) == "http":
                    # HTTP通信方式
                    from dubhe_sdk.http_server import app, setModel
                    setModel(value=model_infos)
                    ThreadPoolExecutor(1).submit(open_browser)
                    app.run(host=IP, port=PORT, threaded=THREADED)
                else:
                    # TCP通信方式
                    from dubhe_sdk.tcp_server import TcpHandler, setModel
                    setModel(value=model_infos)
                    ThreadPoolExecutor(1).submit(connect_tcp, IP, PORT)
                    server = socketserver.ThreadingTCPServer((IP, PORT), TcpHandler)
                    server.serve_forever()
                serve_flag = True
        if serve_flag == False:
            raise Exception('Serve not found!')

    except Exception as ex:
        traceback.print_exc()
        exceptionMSG = traceback.format_exc()
        if TASK_TYPE != TASK_INFERENCE_TYPE and PLATFORM_TYPE == AI_PLATFORM:
            end_json = train_exception_end_data('%s current model error %s' % (INSTANCE_ID, ex))
            send_kafka(MODEL_STATUS, end_json, TOPIC_MODEL_STATUS)
        logger.error(exceptionMSG)


def _find_pipelines(filename: str) -> typing.List[typing.Tuple]:
    pipelines = []

    assert filename[-3:] == '.py' and os.path.isfile(filename), 'Python entry point is not correct'

    _, module_name_with_ext = os.path.split(filename)
    module_name, _ = os.path.splitext(module_name_with_ext)
    module_spec = importlib.util.spec_from_file_location(module_name, filename)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    for name in dir(module):
        if name.startswith('__'):
            continue

        obj = module.__getattribute__(name)
        if hasattr(obj, '_is_pipeline_func'):
            pipelines.append((module, name))
    return pipelines

def _find_entrances(pipelines: typing.List[typing.Tuple]) -> typing.List[typing.Tuple]:
    entrances = []

    for pipe in pipelines:
        module = pipe[0]
        name = pipe[1]
        obj = module.__getattribute__(name)
        entrance = None
        for _, function in inspect.getmembers(
                obj,
                predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x),
        ):
            func_type = None
            if hasattr(function, "_is_train_func"):
                func_type = TASK_TRAIN_TYPE
            elif hasattr(function, "_is_predict_func"):
                func_type = TASK_PREDICT_TYPE
            elif hasattr(function, '_is_inference_func'):
                func_type = TASK_INFERENCE_TYPE

            if func_type is not None:
                entrance = (module, name, func_type, function)
                entrances.append(entrance)

    return entrances

# def taskType():
#     curType = INSTANCE_ID.split('-')[2]
#     if curType == PREFIX_TRAIN:
#         return TASK_TRAIN_TYPE
#     elif curType == PREFIX_PREDICT:
#         return TASK_PREDICT_TYPE
#     elif curType == PREFIX_INFERENCE or curType == PREFIX_AUTOMARK:
#         return TASK_INFERENCE_TYPE
#     else:
#         logger.error("TASK %s PREFIX not defined!"%curType)
#         return -1