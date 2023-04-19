import os
import torch
from dataset import CIFAR10
from network import ConvNet
import tqdm
import torch.optim as optim
from util import evaluate, AverageMeter
import argparse
from torch.utils.tensorboard import SummaryWriter


def MyCELoss(pred, gt):
    # ----------TODO------------
    # Implement CE loss here
    # ----------TODO------------
    pred = torch.exp(pred)
    sum = torch.sum(pred, axis=1)
    pred = (pred.T / sum).T
    a = list(range(pred.shape[0]))
    loss = -torch.sum(torch.log(pred[a, gt])) / pred.shape[0]
    return loss


def validate(epoch, model, val_loader, writer):
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for imgs, labels in tqdm.tqdm(val_loader):
        if torch.cuda.is_available():
            imgs = imgs.cuda()
        bsz = labels.shape[0]
        output = model(imgs)
        if torch.cuda.is_available():
            output = output.cpu()
        # update metric
        acc1, acc5 = evaluate(output, labels, topk=(1, 5))
        top1.update(acc1.item(), bsz)
        top5.update(acc5.item(), bsz)

    # ----------TODO------------
    # draw accuracy curve!
    # ----------TODO------------
    writer.add_scalar('val/top1', top1.avg, epoch)
    writer.add_scalar('val/top5', top5.avg, epoch)

    print(' Val Acc@1 {top1.avg:.3f}'.format(top1=top1))
    print(' Val Acc@5 {top5.avg:.3f}'.format(top5=top5))
    return


def validate_c(epoch, model, val_loader, writer):
    top1 = []
    top5 = []
    for i in range(3):
        model[i].eval()
        top1.append(AverageMeter())
        top5.append(AverageMeter())
    for imgs, labels in tqdm.tqdm(val_loader):
        if torch.cuda.is_available():
            imgs = imgs.cuda()
        bsz = labels.shape[0]
        for i in range(3):
            output = model[i](imgs)
            if torch.cuda.is_available():
                output = output.cpu()
            # update metric
            acc1, acc5 = evaluate(output, labels, topk=(1, 5))
            top1[i].update(acc1.item(), bsz)
            top5[i].update(acc5.item(), bsz)

    # ----------TODO------------
    # draw accuracy curve!
    # ----------TODO------------
    writer.add_scalars('val/top1', {
        '1e-3': top1[0].avg,
        '1e-4': top1[1].avg,
        '1e-5': top1[2].avg
    }, epoch)
    writer.add_scalars('val/top5', {
        '1e-3': top5[0].avg,
        '1e-4': top5[1].avg,
        '1e-5': top5[2].avg
    }, epoch)
    return


def train(epoch, model, optimizer, train_loader, writer):
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    iteration = len(train_loader) * epoch
    for imgs, labels in tqdm.tqdm(train_loader):
        bsz = labels.shape[0]

        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()

        output = model(imgs)
        loss = MyCELoss(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = evaluate(output, labels, topk=(1, 5))
        top1.update(acc1.item(), bsz)
        top5.update(acc5.item(), bsz)

        loss.backward()
        optimizer.step()

        iteration += 1
        if iteration % 50 == 0:
            # pass
            # ----------TODO------------
            # draw loss curve and accuracy curve!
            # ----------TODO------------
            writer.add_scalar('train/loss', loss, iteration)
            writer.add_scalar('train/top1', top1.avg, iteration)
            writer.add_scalar('train/top5', top5.avg, iteration)

    print(' Epoch: %d' % (epoch))
    print(' Train Acc@1 {top1.avg:.3f}'.format(top1=top1))
    print(' Train Acc@5 {top5.avg:.3f}'.format(top5=top5))
    return


def train_c(epoch, model, optimizer, train_loader, writer):
    for i in range(3):
        model[i].train()

    losses = []
    loss_lst = [0, 0, 0]
    top1 = []
    top5 = []
    for i in range(3):
        losses.append(AverageMeter())
        top1.append(AverageMeter())
        top5.append(AverageMeter())
    iteration = len(train_loader) * epoch
    for imgs, labels in tqdm.tqdm(train_loader):
        bsz = labels.shape[0]

        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()

        for i in range(3):
            optimizer[i].zero_grad()

            output = model[i](imgs)
            loss = MyCELoss(output, labels)

            # update metric
            loss_lst[i] = loss
            losses[i].update(loss.item(), bsz)
            acc1, acc5 = evaluate(output, labels, topk=(1, 5))
            top1[i].update(acc1.item(), bsz)
            top5[i].update(acc5.item(), bsz)

            loss.backward()
            optimizer[i].step()

        iteration += 1
        if iteration % 50 == 0:
            # pass
            # ----------TODO------------
            # draw loss curve and accuracy curve!
            # ----------TODO------------
            writer.add_scalars('train/loss', {
                '1e-3': loss_lst[0],
                '1e-4': loss_lst[1],
                '1e-5': loss_lst[2]
            }, iteration)
            writer.add_scalars('train/top1', {
                '1e-3': top1[0].avg,
                '1e-4': top1[1].avg,
                '1e-5': top1[2].avg
            }, iteration)
            writer.add_scalars('train/top5', {
                '1e-3': top5[0].avg,
                '1e-4': top5[1].avg,
                '1e-5': top5[2].avg
            }, iteration)
    return


def run(args):
    save_folder = os.path.join('\experiments', args.exp_name)
    ckpt_folder = os.path.join(save_folder, 'ckpt')
    log_folder = os.path.join(save_folder, 'log')
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)
    writer = SummaryWriter(log_dir=log_folder)

    # define dataset and dataloader
    train_dataset = CIFAR10()
    val_dataset = CIFAR10(train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batchsize,
                                               shuffle=True,
                                               num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batchsize,
                                             shuffle=False,
                                             num_workers=2)

    # define network
    model = ConvNet()
    if torch.cuda.is_available():
        model = model.cuda()

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.cont:
        # load latest checkpoint
        ckpt_lst = os.listdir(ckpt_folder)
        ckpt_lst.sort(key=lambda x: int(x.split('_')[-1]))
        read_path = os.path.join(ckpt_folder, ckpt_lst[-1])
        print('load checkpoint from %s' % (read_path))
        checkpoint = torch.load(read_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.total_epoch):
        train(epoch, model, optimizer, train_loader, writer)

        if epoch % args.save_freq == 0:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(ckpt_folder,
                                     'ckpt_epoch_%s' % (str(epoch)))
            torch.save(state, save_file)

        with torch.no_grad():
            validate(epoch, model, val_loader, writer)
    return


def run_c(args):
    save_folder = os.path.join('\experiments', args.exp_name)
    ckpt_folder = os.path.join(save_folder, 'ckpt')
    log_folder = os.path.join(save_folder, 'log')
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)
    writer = SummaryWriter(log_dir=log_folder)

    # define dataset and dataloader
    train_dataset = CIFAR10()
    val_dataset = CIFAR10(train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batchsize,
                                               shuffle=True,
                                               num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batchsize,
                                             shuffle=False,
                                             num_workers=2)

    lr_lst = [1e-3, 1e-4, 1e-5]
    model = []
    optimizer = []
    for i in range(3):
        # define network
        model.append(ConvNet())
        if torch.cuda.is_available():
            model[i] = model.cuda()
        # define optimizer
        optimizer.append(optim.Adam(model[i].parameters(), lr=lr_lst[i]))

    start_epoch = 0
    total_epoch = 10

    for epoch in range(start_epoch, total_epoch):
        train_c(epoch, model, optimizer, train_loader, writer)
        if epoch % args.save_freq == 0:
            state = {
                'model': model[i].state_dict(),
                'optimizer': optimizer[i].state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(ckpt_folder,
                                     'ckpt_epoch_%s' % (str(epoch)))
            torch.save(state, save_file)
        with torch.no_grad():
            validate_c(epoch, model, val_loader, writer)
    return


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--exp_name',
        '-e',
        type=str,
        required=True,
        help="The checkpoints and logs will be save in ./checkpoint/$EXP_NAME")
    arg_parser.add_argument('--lr',
                            '-l',
                            type=float,
                            default=5e-4,
                            help="Learning rate")
    arg_parser.add_argument('--save_freq',
                            '-s',
                            type=int,
                            default=1,
                            help="frequency of saving model")
    arg_parser.add_argument('--total_epoch',
                            '-t',
                            type=int,
                            default=20,
                            help="total epoch number for training")
    arg_parser.add_argument(
        '--cont',
        '-c',
        action='store_true',
        help=
        "whether to load saved checkpoints from $EXP_NAME and continue training"
    )
    arg_parser.add_argument('--batchsize',
                            '-b',
                            type=int,
                            default=20,
                            help="batch size")
    args = arg_parser.parse_args()

    run(args)
    # run_c(args)  # 对应2(c)，分别使用三个学习率进行训练，并绘制结果图
