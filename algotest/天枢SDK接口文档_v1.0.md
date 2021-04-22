# dubhe-sdk
## 设计目标
格创AI天枢系统算法SDK。

## 适配范围
为了使算法工程师快速将自己的算法上线到格创天枢平台，用户可以通过安装dubhe-sdk，从而调用封装好的接口，获取http服务或tcp服务。平台与算法接口文档https://qwps.com/join/g9kehq0。

## 安装SDK

`pip install dubhe-sdk`

## 使用SDK

`import dubhe_sdk`

## http 接口获取

1. config提供了一些基础配置，它会获取平台所需要的环境变量和kafka等已经约定好的配置。
2. 引入app实例，可以获取到适应平台的http接口。app实例是一个Flask实例，它提供了心跳接口，模型训练接口，模型评估接口，在线推理接口。
3. 你需要通过继承Model类，实现一个自己的模型，重写train、predict、inference方法来提供算法对应的模型训练接口，模型评估接口，在线推理接口。为了让app实例能够获取到你的模型，需要调用set_model方法。
4. 重写train、predict、inference方法时，通过input_params，你可以获取到一个字典类型的平台的http请求参数。
5. 调用`app.run(host=IP, port=PORT, threaded=THREADED)`方法就可以向平台提供所需的http接口。

```python
from dubhe_sdk.config import *
from dubhe_sdk.http_server import app, Model, set_model

class My_Model(Model):
    def __init__(self):
        super().__init__()

    def train(self, input_params):
        pass # 此处重写模型训练算法
    
    def predict(self, input_params):
        pass # 此处重写模型评估算法
    
    def inference(self, input_params):
        pass # 此处重写模型在线推理算法
    
if __name__ == '__main__':
    my_model = My_Model()
    set_model('name',my_model)
	
    app.run(host=IP, port=PORT, threaded=THREADED)
```



## 入参

1. sdk提供了TrainParam、PredictParam、InferenceParam类，他们是可供参考的标准入参格式。你可以使用input_params在你重写的方法中构造好他们，以便使用。
2. 你可以决定你的算法有哪些超参，你需要将你的超参放在config目录的json文件中，这样平台可以加载出来。构造TrainParam类的时候也可以将config文件里面的参数结合一并构造。

```python
from dubhe_sdk.param.train_param import TrainParam
from dubhe_sdk.utils import load_config

def train(self, input_params):
    # 入参与config参数结合
    params = TrainParam._default_values()
    config_params = load_config('config/model_train_config.json')
    self.config_dict_replace(params, config_params)
    self.input_dict_replace(params, input_params)
    logger.info('_params\n')
    logger.info(self.params)

    self.train_params = TrainParam()
    self.train_params.__dict__ = params
```

## kafka

- 算法启动成功提供http服务后需要向平台kafka发送READY消息，http_server中已经封装好了，这一步你无需做任何操作。
- 算法启动需要向平台kafka发送START消息，调用这两句就可以发送。

```python
from dubhe_sdk.pipeline.ADCkafka import *
start_json = train_start_data()
send_kafka(MES_TRAIN_STATUS, start_json, TOPIC_MODEL_STATUS, os.path.join(self.resultDir, 'json.dat'))
```

- 算法训练进行过程中需要向平台kafka发送进度消息，调用这两句就可以发送。

```python
# progress, epoch, loss, lr, acc
progress_json = train_progress_data(progress, epoch, loss.cpu().item(), args.lr, acc.cpu().item(), -1, -1)
send_kafka(MES_TRAIN_PROGRESS, progress_json, TOPIC_MODEL_STATUS,
           os.path.join(os.path.dirname(args.save_path), 'results', 'json.dat'))
```

- 算法正常训练结束需要向平台kafka发送END消息，调用这两句就可以发送。

```python
end_json = train_end_data(model_files)
send_kafka(MES_TRAIN_STATUS, end_json, TOPIC_MODEL_STATUS, os.path.join(self.resultDir, 'json.dat'))
```

- 算法异常训练结束需要向平台kafka发送END消息，调用这两句就可以发送。

```python
end_json = train_exception_end_data(msg)
send_kafka(MES_TRAIN_STATUS, end_json, TOPIC_MODEL_STATUS, os.path.join(self.resultDir, 'json.dat'))
```



## 中间结果和模型输出

- sdk提供Model类的localDirs用来保存中间结果，outputModelDir用来保存模型结果。模型训练前你需要创建这些目录。

```python
# prepeare dir
for tmpDir in self.localDirs:
    if not os.path.exists(tmpDir):
        os.makedirs(tmpDir)
if not os.path.exists(self.outputModelDir):
    os.makedirs(self.outputModelDir)
```

- 模型文件保存需要按照这个格式：model + 毫秒级时间戳 + epoch + loss.pb - 模型文件。

```python
save_model_epoch = 'model_{}_{}_{:.2f}.pth'.format(get_time_stamp(), epoch, loss.cpu().item())
torch.save(model.state_dict(), os.path.join(self.outputModelDir, save_model_epoch)
          )
```



- 算法日志保存。

```python
from dubhe_sdk.pipeline.Logger import ADCLog
self.logfile = os.path.join(self.logsDir, 'log.txt')
self.logger = ADCLog.GetLogger(self.logfile)
self.logger.info('split train and test starting...')
```











