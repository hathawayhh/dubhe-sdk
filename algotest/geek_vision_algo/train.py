import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.utils.data.dataset
import time
import json

from torch.autograd import Variable

from torch.optim.lr_scheduler import *
import geek_vision_algo.model as mm

def evall(str):
    return eval(str)

def get_time_stamp():
    ct = time.time()
    localtime = time.localtime(ct)
    data_head = time.strftime("%Y%m%d%H%M%S", localtime)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s%03d" % (data_head, data_secs)
    return time_stamp

def main(argv, train_set, test_set, ctx):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='resnet18', help="choose model")
    parser.add_argument("--init_epoch", type=int, default=0, help="start epoch")
    parser.add_argument("--epochs", type=int, default=10000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--pretrained_weights", type=str,help="if specified starts from checkpoint model")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--save_path", type=str, default='models/weights_1350_0102', help="save model path")
    # progress_init+progress_ratio*train_progress
    parser.add_argument("--progress_init", type=float, default=0.0, help="progress init")
    parser.add_argument("--progress_ratio", type=float, default=1.0, help="progress ratio")
    parser.add_argument("--model_optimizer", type=str, default='', help="choose optimizer")
    parser.add_argument("--code_list", nargs='*', help="code list to train")
    parser.add_argument("--model_num", type=int, default=0,help="choose model")
    args = parser.parse_args(argv)
    # print_args(args)
    args.class_num = len(args.code_list)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    resnet = getattr(mm, args.model_name)()
    model = mm.Net(resnet, args.model_name, args.class_num)
    if torch.cuda.is_available():
        model.cuda()
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
        ctx.log("\nEpoch: %d" % epoch)
        model.train()
        train_acc = 0.0
        for batch_idx, (_, img, label) in enumerate(train_loader):
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
            train_acc = num_correct / num_total

            progress = ((batch_idx + 1) / len(train_loader) + epoch)*args.progress_ratio / args.epochs
            Metrics = dict()
            Metrics['loss'] = round(loss.item(), 6)
            Metrics['acc'] = train_acc
            ctx.logProgressByBatch(epoch, batch_idx, len(train_loader), progress, Metrics)

        if epoch % args.evaluation_interval == 0 or epoch == args.epochs-1:
            ctx.log("\nValidation Epoch: %d" % epoch)
            model.eval()
            total = 0
            correct = 0
            with torch.no_grad():
                for valid_i, (_, img, label) in enumerate(valid_loader):
                    image = Variable(img.cuda())
                    label = Variable(label.cuda())
                    out = model(image)
                    _, predicted = torch.max(out.data, 1)

                    total += image.size(0)
                    correct += predicted.data.eq(label.data).cpu().sum()

            save_model_epoch = 'model_{}_{}_{:.2f}.pth'.format(get_time_stamp(), epoch, loss.cpu().item())
            state = {
                "code_list": args.code_list,
                "state_dict": model.state_dict()
            }
            torch.save(state, os.path.join(args.save_path, save_model_epoch))
            ctx.log('Snapshot saved to %s' % save_model_epoch)
        scheduler.step()

        # progress = (epoch + 1) / (args.epochs + 1)
        # Metrics = dict()
        # Metrics['loss'] = round(loss.item(),6)
        # Metrics['accuracy'] = train_acc
        # ctx.logProgressByEpoch(epoch, round(progress, 2), Metrics)



if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv[1:])