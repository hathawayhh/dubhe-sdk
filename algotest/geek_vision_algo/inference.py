# -*- coding: UTF-8 -*-
import sys
import argparse
# from utils import print_args
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split
from torch.autograd import Variable
import numpy as np
import cv2
from PIL import Image

def predict_classify_details(img_path, ground_true, predict_code, conf):
    """
    {
		"img_path": "img_full_path"  ,
		"ground_true": "code1" # ,
		"predict_code": "code2",
		"conf": 0.8 # 0-1 float ,
	}
    :return:
    """
    data_list = []
    data = {}
    data['ground_true'] = ground_true
    data['predict_code'] = predict_code
    data['conf'] = conf
    data_list.append(data)

    result = {}
    result['img_path'] = img_path
    result['info'] = data_list
    return [result]

def main(argv, test_set, model, ctx):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_txt", type=str)
    parser.add_argument("--output_img", type=str)
    parser.add_argument("--conf_thres", type=float)
    parser.add_argument("--iou_thres", type=float)
    parser.add_argument("--batch_size", type=int,default=4, help="size of each image batch")
    parser.add_argument("--progress_init", type=float, default=0.0, help="progress init")
    parser.add_argument("--progress_ratio", type=float, default=1.0, help="progress ratio")
    parser.add_argument("--code_list", nargs='*')
    parser.add_argument("--model_num", type=int, default=0, help="choose model")
    args = parser.parse_args(argv)
    args.class_num = len(args.code_list)
    # print_args(args)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # endregion
    progress_init = 0.0
    progress_ratio = 1.0
    data = []
    with open(args.output_txt, 'w') as ftxt:
        for index,(imgPath, image, classname) in enumerate(test_loader):
            image = Variable(image.cuda(), requires_grad=False)
            classname = Variable(classname.cuda())
            with torch.no_grad():
                out = model(image)
                out = torch.sigmoid(out).detach().cpu()
                _, predicted = torch.max(out, 1)
                labels = [args.code_list[i] for i in predicted]
                classnames = [args.code_list[i] for i in classname]
                conf = out[0,predicted]
                for i in range(len(labels)):
                    ftxt.write('{} {} {} {}\n'.format(imgPath[i], classnames[i], labels[i],float(conf[i])))
                    result = ctx.logResultDetails(imgPath[i], classnames[i], labels[i], round(float(conf[i]),6))
                    # predict_json = predict_classify_details(imgPath[i], classnames[i], labels[i], round(float(conf[i]),6))
                    # ctx.logResult(predict_json)
                    data.extend(result)
    return data



if __name__ == '__main__':
    print(sys.argv)
    main(sys.argv[1:])
