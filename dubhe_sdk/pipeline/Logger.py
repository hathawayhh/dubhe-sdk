
import logging
from logging.handlers import RotatingFileHandler


class ADCLog(object):
    _logger = None

    _main_logger = None

    @classmethod
    def getDefaultLogger(cls):
        if ADCLog._logger is None:
            return ADCLog.GetLogger('output/log.txt')
        else:
            return ADCLog._logger

    @classmethod
    def getMainLogger(cls):
        if ADCLog._main_logger is None:
            logger = logging.getLogger("main_logger")
            if not logger.handlers:
                logfile = 'log.txt'
                formatter = logging.Formatter('[%(asctime)s] [%(filename)s:%(lineno)s]\t%(levelname)s: %(message)s')
                # 控制台输出
                streamHandler = logging.StreamHandler()
                streamHandler.setFormatter(formatter)
                # 滚动文件输出
                log_file_handler = RotatingFileHandler(filename=logfile, mode='a', maxBytes=10 * 1024 * 1024, backupCount=14,encoding=None, delay=0)
                log_file_handler.setFormatter(formatter)

                logger.addHandler(streamHandler)
                logger.addHandler(log_file_handler)
                logger.setLevel(logging.INFO)

            ADCLog._main_logger = logger
        return ADCLog._main_logger

    @classmethod
    def GetLogger(cls, filename):
        logger = logging.getLogger("mylogger")
        if not logger.handlers:
            formatter = logging.Formatter('[%(asctime)s] [%(filename)s:%(lineno)s]\t%(levelname)s: %(message)s')
            # 控制台输出
            streamHandler = logging.StreamHandler()
            streamHandler.setFormatter(formatter)
            # 滚动文件输出
            log_file_handler = RotatingFileHandler(filename=filename, mode='a', maxBytes=10 * 1024 * 1024, backupCount=14,encoding=None, delay=0)
            log_file_handler.setFormatter(formatter)

            logger.addHandler(streamHandler)
            logger.addHandler(log_file_handler)
            logger.setLevel(logging.INFO)
        ADCLog._logger = logger
        return logger
