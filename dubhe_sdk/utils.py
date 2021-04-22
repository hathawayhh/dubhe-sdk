import time
import json
from dubhe_sdk.pipeline.Logger import ADCLog
logger = ADCLog.getMainLogger()

# 获取时间戳；
def get_time_stamp():
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y%m%d%H%M%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s%03d" % (data_head, data_secs)
    return time_stamp

# 打印参数
def print_args(args):
    print()
    for key, value in vars(args).items():
        if value is None:
            value = 'None'
        print('{:<50}{}'.format(key, value))
    print('\n')

# 加载json
def load_config(config_file):
    try:
        with open(config_file) as file:
            config = json.load(file)
            return config
    except OSError as e:
        logger.info(e)
        return None

if __name__ == "__main__":
    print(get_time_stamp())