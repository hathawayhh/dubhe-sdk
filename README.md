

# dubhe-sdk

## 设计目标
格创AI天枢系统算法SDK。

## 适配范围
为了使算法工程师快速将自己的算法上线到格创天枢平台，用户可以通过安装dubhe-sdk，从而调用封装好的接口，获取http服务或tcp服务。天枢平台与算法接口文档https://qwps.com/join/g9kehq0。

## 安装SDK

`pip install dubhe-sdk`

## 使用SDK

`import dubhe_sdk`

## 接口获取

1. **通信协议支持**

   天枢平台支持两种通信协议，http协议或tcp协议。在天枢平台新建算法时需要指定使用的接口类型。SDK支持这两种接口，但是需要你在run.sh中自定义指定protocol的环境变量。

2. **启动执行文件**

   python启动执行文件默认为app.py，你需要在app.py导入dubhe_sdk包，SDK才可以在启动时就为你封装好接口。

3. **接口支持**

   SDK目前支持天枢平台与算法之间的四种接口：心跳检测接口、模型训练接口、模型评估接口、在线推理接口。使用`@dubhe_sdk.pipeline`装饰你自定义的模型类，使用`@dubhe_sdk.train`装饰自定义的train方法，可以获取到模型训练的接口，使用`@dubhe_sdk.predict`装饰自定义的predict方法，可以获取到离线评估的接口，使用`@dubhe_sdk.inference`装饰自定义的inference方法，可以获取到在线推理的接口。

4. **启动加载项**

   你自定义的模型类如果需要在算法启动的时候提前加载好自定义的config配置，或预训练模型，或推理模型等，可以在`__init__`方法中进行定义。

5. **入参**

   `__init__`方法、train方法、predict方法、inference方法必须定义参数ctx，其类型是dubhe_sdk.Context类。使用ctx可以获取到所需的环境变量及请求入参。SDK将所有的参数封装在dubhe_sdk.ContextKey类中。

   调用`ctx.get(dubhe_sdk.ContextKey.ENV_XXX)`可以获取到环境变量，调用`ctx.get(dubhe_sdk.ContextKey.KEY_XXX)`可以获取到请求入参。

使用示例：

run.sh

```shell
export protocol="tcp"
python3 -u app.py
```

app.py

```python
import dubhe_sdk

@dubhe_sdk.pipeline()
class Predict_Model():
    def __init__(self, ctx:dubhe_sdk.Context):
        config_params = load_config('config/model_predict_config.json')
        self.model_name = config_params['model']
        self.batch_size = config_params['batch_size']
        # 离线评估启动加载模型
        model_path = ctx.get(dubhe_sdk.ContextKey.ENV_MODEL_PATH)
        model_dict = torch.load(model_path)
        self.code_list = model_dict['code_list']
        self.state_dict = model_dict['state_dict']

    @dubhe_sdk.predict()
    def predict(self, ctx: dubhe_sdk.ContextKey):
        self.data_paths = ctx.get(dubhe_sdk.ContextKey.KEY_DATA_PATHS)
        self.debug_dir = ctx.get(dubhe_sdk.ContextKey.KEY_TMP_DIR)
```



## kafka

1. **算法服务启动支持**

   算法启动完端口，成功提供算法服务后，SDK向平台kafka发送READY状态消息，通知平台服务可用。

2. **模型开始/结束支持**

   算法接收到tcp/http请求后，模型训练/离线评估等耗时任务进入异步线程，SDK向平台kafka发送START消息，通知平台模型训练/离线评估任务开始。当任务正常结束，SDK向平台kafka发送END消息，通知平台任务结束。当任务异常结束，SDK捕获异常信息，向平台kafka发送带有异常信息的END消息。

3. **模型进度/指标支持**

   - by epoch 发送模型进度/指标

     方法：`dubhe_sdk.Context.logProgressByEpoch`

     解释：用于模型训练时，by epoch发送进度与指标。

     使用示例：

     ```python
     progress = (epoch + 1) / (args.epochs + 1)
     Metrics = dict()
     Metrics['loss'] = round(loss.mean().item(),6)
     Metrics['accuracy'] = train_acc
     ctx.logProgressByEpoch(epoch, round(progress, 2), Metrics)
     ```

   - by batch 发送模型进度/指标

     方法：`dubhe_sdk.Context.logProgressByBatch`

     解释：用于模型训练时，每个epoch耗时较长，可以by batch发送进度与指标。

     使用示例：

     ```python
     progress = ((batch_idx + 1) / len(train_loader) + epoch)*args.progress_ratio / args.epochs
     Metrics = dict()
     Metrics['loss'] = round(loss.item(), 6)
     Metrics['accuracy'] = train_acc
     ctx.logProgressByBatch(epoch, batch_idx, len(train_loader), progress, Metrics)
     ```

     Metrics需要传入一个指标名与指标值为键值对的字典，平台接受到后会按指标分别绘制曲线。

   - 只发送模型进度

     方法：`dubhe_sdk.Context.logProgress`

     解释：用于模型训练/离线评估时，仅更新模型进度。

     使用示例：
     
     ```
     ctx.logProgress(progress)
     ```
     
     

4. **验证集/评估集预测结果支持**

   算法在以下两种场景下需要向平台kafka返回预测结果消息，以便平台对模型效果进行查看：

   场景1：模型训练完成后，需要将模型应用于验证集进行预测评估，并向kafka返回验证集的预测结果；

   场景2：模型离线评估时，需要将评估集预测结果返回kafka。

   

   预测结果消息需要遵循平台的json格式：

   - 按照平台规范对结果进行了固定json格式的封装，需要传入具体参数。

     方法：`dubhe_sdk.Context.logResultDetails`

     解释：用于模型训练/离线评估时，向kafka返回预测结果；或用于在线推理时，用于log记录与返回预测结果json。
   
     > 注: 在线推理由于消息的实时性要求，直接通过tcp/http接口返回预测结果，无需发送kafka。

     使用示例：

     ```python
     ctx.logResultDetails(imgPath, classnames, labels, round(float(conf),6))
     ```

     
   
   - 需要自己按照平台固定json格式封装结果，可以增加自定义结果进行返回。
   
     解释：用于模型训练/离线评估时，向kafka返回预测结果；或用于在线推理时，仅作为log记录。
   
     使用示例：
     
     ```python
     ctx.logResult(predict_json)
     ```



## 中间结果和模型输出

- SDK提供KEY_OUTPUT_DIR用来保存训练生成的模型文件，KEY_TMP_DIR用来保存模型的中间结果。模型任务开始前你可以根据需要创建这些目录。

  使用示例：

  ```python
  self.output_dir = ctx.get(dubhe_sdk.ContextKey.KEY_OUTPUT_DIR)
  self.debug_dir = ctx.get(dubhe_sdk.ContextKey.KEY_TMP_DIR)
  # prepeare dir
  if not os.path.exists(self.debug_dir):
      os.makedirs(self.debug_dir, exist_ok=True)
      if not os.path.exists(self.output_dir):
          os.makedirs(self.output_dir, exist_ok=True)
  ```

  

- 模型文件保存需要按照这个格式：`model_毫秒级时间戳_epoch_loss.文件名后缀`，天枢平台默认将最后一个epoch设置为最优模型。

  使用示例：

  ```python
  save_model_epoch = 'model_{}_{}_{:.2f}.pth'.format(get_time_stamp(), epoch, loss.cpu().item())
  torch.save(model.state_dict(), os.path.join(self.output_dir, save_model_epoch))
  ```

  

- 算法日志保存。

  SDK产出的日志会保存在容器的`/temporary/log` 的dubhe.log文件中，你可以调用`ctx.log`方法来输出log。也可以使用自定义的log模块。

  使用示例：

  ```python
  ctx.log('model train starting...')
  ```
