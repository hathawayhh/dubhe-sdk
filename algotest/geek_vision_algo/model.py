import torch.nn as nn
from torchvision.models.resnet import resnet18
from torchvision.models.resnet import resnet34
from torchvision.models.resnet import resnet50
from torchvision.models.resnet import resnet101
from torchvision.models.resnet import resnet152
from torchvision.models.resnet import resnext50_32x4d
from torchvision.models.resnet import resnext101_32x8d
from torchvision.models.resnet import wide_resnet50_2
from torchvision.models.resnet import wide_resnet101_2
from torchvision.models.vgg import vgg16
from torchvision.models.vgg import vgg19
from torchvision.models.alexnet import alexnet

model_linear = [512, 512, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 25088, 25088, 9216]

class Net(nn.Module):
    def __init__(self, model, args):
        super(Net, self).__init__()
        # 取掉model的后1层
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        self.Linear_layer = nn.Linear(model_linear[args.model_num], args.class_num)  # 加上一层参数修改好的全连接层512

    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer(x)
        return x