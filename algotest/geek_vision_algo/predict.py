# -*- coding: UTF-8 -*-
import sys
import argparse
# from utils import print_args
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split
from geek_vision_algo.model import *
from dubhe_sdk.pipeline.ADCkafka import *
from torch.autograd import Variable
import numpy as np
import cv2
from PIL import Image

def main(argv, test_set):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str)
    parser.add_argument("--output_txt", type=str)
    parser.add_argument("--output_img", type=str)
    parser.add_argument("--conf_thres", type=float)
    parser.add_argument("--iou_thres", type=float)
    parser.add_argument("--predict_method", type=str, default='predict', choices=['predict', 'train'])
    parser.add_argument("--progress_init", type=float, default=0.0, help="progress init")
    parser.add_argument("--progress_ratio", type=float, default=1.0, help="progress ratio")
    parser.add_argument("--code_list", nargs='*')
    parser.add_argument("--model_num", type=int, default=0, help="choose model")
    args = parser.parse_args(argv)
    args.class_num = len(args.code_list)
    # print_args(args)

    model_dict = torch.load(args.pretrained_weights)

    resnet = resnet18(pretrained=False)
    model = Net(resnet, args)
    model = model.to('cuda')

    model.load_state_dict(model_dict)
    model.eval()

    test_files = np.array(test_set.dataset.imgs)[np.array(test_set.indices)]

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0,
                               hue=0),
        transforms.ToTensor(),
        normalize,
    ])

    # region please design your predict logic here
    '''
    send progress to kafka
    Example::
    >>> if predict_method == 'predict':
    ...   progress = progress_init + progress * progress_ratio
    ...   progress_json = predict_process_data(progress)
    ...   send_kafka(MES_PREDICT_PROGRESS, progress_json, TOPIC_MODEL_STATUS, os.path.join(os.path.dirname(result_txt), 'json.dat'))

    '''

    '''
    send predict result to kafka
    Example::
    >>> if predict_method=='predict':
    ...   send_kafka(MES_RESULT_INFERENCE_CLASSIFY, predict_json, TOPIC_MODEL_RESULT, os.path.join(os.path.dirname(result_txt), 'json.dat'))
    ... else:
    ...   send_kafka(MES_RESULT_TRAIN_CLASSIFY, predict_json, TOPIC_MODEL_RESULT,os.path.join(os.path.dirname(result_txt), 'json.dat'))
    '''
    # endregion

    total_img = len(test_files)
    progress_init = 0.0
    progress_ratio = 1.0
    with open(args.output_txt, 'w') as ftxt:
        for i,(image_file, classname) in enumerate(test_files):
            img = Image.open(image_file)
            assert img is not None, image_file
            img_t = transform(img)
            inputs = img_t.unsqueeze(0)
            with torch.no_grad():
                inputs = Variable(inputs.to('cuda'), requires_grad=False)
                out = model(inputs)
                out = torch.sigmoid(out).detach().cpu()
                _, predicted = torch.max(out, 1)
                label = args.code_list[predicted]
                conf = out[0,predicted]
                ftxt.write('{} {} {} {}\n'.format(image_file, args.code_list[int(classname)], label, conf))
                predict_json = predict_classify_details(image_file, args.code_list[int(classname)], label, float(conf))

            # send to kafka
            if args.predict_method == 'predict' and ((i + 1) % 100 == 0 or (i + 1) == total_img):
                progress = (i + 1) / total_img
                progress = progress_init + progress * progress_ratio
                progress_json = predict_process_data(progress)
                send_kafka(MES_PREDICT_PROGRESS, progress_json, TOPIC_MODEL_STATUS,
                           os.path.join(os.path.dirname(args.output_txt), 'json.dat'))
            if args.predict_method == 'predict':
                send_kafka(MES_RESULT_INFERENCE_CLASSIFY, predict_json, TOPIC_MODEL_RESULT,
                           os.path.join(os.path.dirname(args.output_txt), 'json.dat'))
            else:
                send_kafka(MES_RESULT_TRAIN_CLASSIFY, predict_json, TOPIC_MODEL_RESULT,
                           os.path.join(os.path.dirname(args.output_txt), 'json.dat'))


if __name__ == '__main__':
    print(sys.argv)
    main(sys.argv[1:])
