from socket import *
import time
import json
import struct
import toml
import os

back_log = 5
buffer_size = 2048

def load_toml_config(config_file):
    try:
        with open(config_file,encoding='utf-8') as f:
            config = toml.load(f)
            return config
    except OSError as e:
        print(e)
        return None

def load_config(config_file):
    try:
        with open(config_file, encoding='utf-8') as f:
            config = json.load(f)
            return config
    except OSError as e:
        print(e)
        return None

def log(str):
    t = time.strftime(r"%Y-%m-%d %H:%M:%S", time.localtime())
    print("[%s]%s" % (t, str))

class Image(object):
    def __init__(self, dir, filename, gt):
        self.IMAGEDIR = dir
        self.NAME = filename
        self.GT = gt

def gen_json(images):  # images in same folder
    with open("inference.json", encoding='utf-8') as file:
        imgdir = images[0].IMAGEDIR
        # JSON_NAME = '%s.json' % (imgdir)
        msg = json.load(file)
        msg["image_base_dir"] = imgdir
        if msg:
            image_names = []
            img_params = []
        #
            for image in images:
        #         img_param = {}
        #         img_param["strHEIGHT"] = ""
        #         img_param["strRATION"] = 10
        #         img_param["strDEFECTSIZE"] = "10"
        #         img_param["strNAME"] = image.NAME
        #         img_param["strGRAYNAME"] = image.GRAYNAME
        #         img_param["GATE1"] = 10
                image_names.append(image.NAME)
        #         img_params.append(img_param)

        msg["image_names"] = image_names
        # msg["special_params"]["img_params"] = img_params

    #with open(JSON_NAME, 'w') as f:
     #   json.dump(msg, f, indent=4)

    return msg

class client():
    def __init__(self):
        self.ready = False
        config = load_toml_config('config.toml')
        # tcp配置
        self.host = config['host']
        self.port = config['port']
        self.req_type = config['req_type']
        self.heart_url = "http://{}:{}/heart".format(self.host,self.port)
        if self.req_type == 0:
            self.request = load_config("heart.json")
        elif self.req_type == 1:
            self.request = load_config("train.json")
        elif self.req_type == 2:
            self.request = load_config("predict.json")
        elif self.req_type == 3:
            self.json_requests = []
            data_all = []
            imgPath = config['dir']
            # iterate all dir to search image
            for path, dirnames, files in os.walk(imgPath):

                for file in files:
                    fullpath = os.path.join(path, file)
                    images_tmp = []
                    if (fullpath.endswith('.jpg') or fullpath.endswith('.JPG')) and ('_G' not in fullpath):
                        path = path.replace('\\', '/')
                        last_dir = path.split('/')[-1]
                        penultimate_dir = path.split('/')[-2]
                        gt = penultimate_dir if penultimate_dir.startswith('T') else last_dir
                        data = Image(path, file, gt)
                        data_all.append(data)
                        images_tmp.append(data)
                        if len(images_tmp) > 0:
                            self.json_requests.append(gen_json(images_tmp))

        else:
            raise ValueError("req_type error")

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


if __name__ == '__main__':
    client = client()
    if client.req_type == 0 or client.req_type == 1 or client.req_type == 2:
        client.send()
    elif client.req_type == 3:
        for request in client.json_requests:
            client.request = request
            client.send()