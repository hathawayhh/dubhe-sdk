import os
import logging
from logging.handlers import RotatingFileHandler

class Logger(object):
    logger = None

    def __init__(self, loglevel = logging.INFO, maxMB = 10, backupCount = 14):
        raise RuntimeError('Call instance() instead')

    @classmethod
    def instance(cls):
        if cls.logger is None:
            #TODO: config to limit logs by days
            path = "/temporary/log"

            if not os.path.exists(path):
                os.makedirs(path)
            filename = '%s/dubhe.log'%path
            formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)s: %(message)s')
            # 控制台输出
            streamHandler = logging.StreamHandler()
            streamHandler.setFormatter(formatter)
            # 滚动文件输出
            log_file_handler = RotatingFileHandler(filename=filename, mode='a', maxBytes=10*1024*1024, backupCount=14, encoding=None, delay=0)
            log_file_handler.setFormatter(formatter)
            cls.logger = logging.getLogger()
            cls.logger.setLevel(logging.INFO)
            cls.logger.addHandler(streamHandler)
            cls.logger.addHandler(log_file_handler)

            streamHandler.close()
            log_file_handler.close()

            # 过滤kafka INFO级别日志
            kafka_logger = logging.getLogger('kafka')
            kafka_logger.setLevel(logging.WARNING)

        return cls.logger