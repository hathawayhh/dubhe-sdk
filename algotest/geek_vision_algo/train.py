import sys
import argparse
from dubhe_sdk.pipeline.ADCkafka import *
# from utils import print_args
from dubhe_sdk.utils import get_time_stamp
import torch
import torch.nn as nn
import torch.utils.data.dataset

from torch.autograd import Variable
from dubhe_sdk.pipeline.Logger import ADCLog
logger = ADCLog.getMainLogger()

from torch.optim.lr_scheduler import *
from geek_vision_algo.model import *

def evall(str):
    return eval(str)

def main(argv, train_set, test_set):
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_epoch", type=int, default=0, help="start epoch")
    parser.add_argument("--epochs", type=int, default=10000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=150, help="size of each image batch")
    parser.add_argument("--pretrained_weights", type=str,help="if specified starts from checkpoint model")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--save_path", type=str, default='models/weights_1350_0102', help="save model path")
    # progress_init+progress_ratio*train_progress
    parser.add_argument("--progress_init", type=float, default=0.0, help="progress init")
    parser.add_argument("--progress_ratio", type=float, default=1.0, help="progress ratio")
    parser.add_argument("--model_optimizer", type=str, default='', help="choose optimizer")
    parser.add_argument("--class_num", type=int, default=2,help="class name number")
    parser.add_argument("--model_num", type=int, default=0,help="choose model")
    args = parser.parse_args(argv)
    # print_args(args)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    resnet = resnet18(pretrained=False)
    model = Net(resnet, args)
    model = model.to('cuda')
    optimizer = ''
    if args.model_optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    if args.model_optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

    scheduler = StepLR(optimizer, step_size=5, gamma=0.95)
    criterion = nn.CrossEntropyLoss()

    # region please design your algorithm training logic here
    '''
    save model weight to model file by evaluation_interval epoch
    模型文件命名规则：model + 毫秒级时间戳 + epoch + loss
    Example::
    >>> for epoch in range(args.init_epoch, args.epochs):
    ...   if epoch % args.evaluation_interval == 0:
    ...      save_model_epoch = 'model_{}_{}_{:.2f}.pth'.format(get_time_stamp(), epoch, loss)
    ...      model.save_weights(os.path.join(args.save_path, save_model_epoch))
    '''
    for epoch in range(args.init_epoch, args.epochs):
        logger.info("\nEpoch: %d" % epoch)
        model.train()
        train_acc = 0.0
        for batch_idx, (img, label) in enumerate(train_loader):
            image = Variable(img.cuda())
            label = Variable(label.cuda())
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            num_total = output.shape[0]
            _, pred_label = output.max(1)
            num_correct = (pred_label == label).sum().item()
            acc = num_correct / num_total
            logger.info("Epoch:%d [%d|%d] loss:%f acc:%f" % (epoch, batch_idx, len(train_loader), loss.mean(), train_acc))

            # send kafka
            progress = ((batch_idx + 1) / len(train_loader) + epoch)*args.progress_ratio / args.epochs
            progress = args.progress_init + progress * args.progress_ratio

            # progress, epoch, loss, lr, acc
            progress_json = train_progress_data(progress, epoch, loss.cpu().item(), args.lr, acc, -1, -1)
            send_kafka(MES_TRAIN_PROGRESS, progress_json, TOPIC_MODEL_STATUS, os.path.join(os.path.dirname(args.save_path), 'results', 'json.dat'))

        if epoch % args.evaluation_interval == 0 or epoch == args.epochs-1:
            logger.info("\nValidation Epoch: %d" % epoch)
            model.eval()
            total = 0
            correct = 0
            with torch.no_grad():
                for valid_i, (img, label) in enumerate(valid_loader):
                    image = Variable(img.cuda())
                    label = Variable(label.cuda())
                    out = model(image)
                    _, predicted = torch.max(out.data, 1)

                    total += image.size(0)
                    correct += predicted.data.eq(label.data).cpu().sum()

            logger.info("Acc: %f " % ((1.0 * correct.numpy()) / total))
            save_model_epoch = 'model_{}_{}_{:.2f}.pth'.format(get_time_stamp(), epoch, loss.cpu().item())
            torch.save(model.state_dict(), os.path.join(args.save_path, save_model_epoch))
            logger.info('Snapshot saved to %s' % save_model_epoch)
        scheduler.step()

    ''' 
    send progress to kafka by epoch
    Example::
    >>> progress = args.progress_init + 1.0 * args.progress_ratio
    >>> progress_json = train_progress_data(progress, epoch, loss, lr, acc, -1, -1)
    >>> send_kafka(MES_TRAIN_PROGRESS, progress_json, TOPIC_MODEL_STATUS, os.path.join(os.path.dirname(args.save_path), 'results', 'json.dat'))
    '''
    # # send kafka
    # progress = args.progress_init + 1.0 * args.progress_ratio
    # # progress, epoch, loss, lr, acc
    # progress_json = train_progress_data(progress, epoch, loss.cpu().item(), args.lr, acc.cpu().item(), -1, -1)
    # send_kafka(MES_TRAIN_PROGRESS, progress_json, TOPIC_MODEL_STATUS,
    #            os.path.join(os.path.dirname(args.save_path), 'results', 'json.dat'))

    # endregion


if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv[1:])