from socket import *
import time
import json
import struct
from concurrent.futures import ThreadPoolExecutor
from kafka import KafkaConsumer
import os

back_log = 5
buffer_size = 2048

def load_config(config_file):
    try:
        with open(config_file,encoding='utf-8') as f:
            config = json.load(f)
            return config
    except OSError as e:
        print(e)
        return None

def log(str):
    t = time.strftime(r"%Y-%m-%d %H:%M:%S", time.localtime())
    print("[%s]%s" % (t, str))

class client():
    def __init__(self):
        self.ready = False
        config = load_config('config.json')
        # 环境变量
        self.PLATFORM_TYPE = 0
        if 'PLATFORM_TYPE' in os.environ and int(os.environ['PLATFORM_TYPE']) == 1:
            self.PLATFORM_TYPE = 1

            # tcp配置
        self.host = config['host']
        self.port = config['port']
        self.heart_url = "http://{}:{}/heart".format(self.host,self.port)
        # kafka配置
        self.BOOTSTRAP_SERVERS = config['BOOTSTRAP_SERVERS']
        self.TOPIC_MODEL_STATUS = config['TOPIC_MODEL_STATUS']
        self.TOPIC_MODEL_RESULT = config['TOPIC_MODEL_RESULT']
        self.request = load_config("train.json")

    def send(self):

        if self.request:
            jmsg = json.dumps(self.request)
            try:
                # 创建Client并连接
                tcp_client = socket(AF_INET, SOCK_STREAM)
                tcp_client.connect((self.host, self.port))
                log("Send: {}".format(jmsg))
                message = jmsg.encode("utf-8")
                sendLen = len(message)
                lenData = struct.pack('!i', sendLen)
                tcp_client.send(lenData)
                tcp_client.sendall(message)

                # 接收message
                head = bytes()
                msgBody = bytes()
                while len(head) < 4:
                    head += tcp_client.recv(4 - len(head))

                msgLen = struct.unpack('!i', head)[0]
                while msgLen > 0:
                    data = tcp_client.recv(buffer_size)
                    msgBody += data
                    msgLen -= len(data)
                jresp = json.loads(msgBody.decode('utf-8'))
                log("Recv: {}" .format(jresp))


            finally:
                tcp_client.close()

    def kafka_recv(self, topic):
        consumer = KafkaConsumer(topic, bootstrap_servers=[self.BOOTSTRAP_SERVERS])
        for msg in consumer:
            recv = "%s:%d:%d: key=%s value=%s" % (msg.topic, msg.partition, msg.offset, msg.key, msg.value)
            log(recv)
            kafka_msg = json.loads(msg.value)
            if kafka_msg['mes_type'] == 0 and kafka_msg['data']['action'] == 'READY':
                self.ready = True

    def kafka(self):
        # kafka接收
        if self.PLATFORM_TYPE == 1:
            ThreadPoolExecutor(1).submit(self.kafka_recv,self.TOPIC_MODEL_STATUS)
            ThreadPoolExecutor(1).submit(self.kafka_recv,self.TOPIC_MODEL_RESULT)
        else:
            self.ready = True

if __name__ == '__main__':
    client = client()
    client.kafka()
    while True:
        if client.ready:
            client.send()
            break