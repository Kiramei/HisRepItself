from utils import amass3d as datasets
from model import AttModel
from utils.opt import Options
from utils import util
from utils import log

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from progress.bar import Bar
import time
import h5py
import torch.optim as optim


def main(opt):
    # 学习率获取
    lr_now = opt.lr_now
    start_epoch = 1
    # opt.is_eval = True
    print('>>> create models')
    # 特征数
    in_features = opt.in_features  # 54
    # 选取的模型
    d_model = opt.d_model
    # 卷积核大小
    kernel_size = opt.kernel_size
    # 跑注意力模型AttModel，加入各种参数
    net_pred = AttModel.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n)
    # 转移到GPU运算
    net_pred.cuda()

    # 利用Adam优化器，定义了一个 Adam 优化器对象 optimizer，使用 filter() 函数过滤出所有需要更新的参数，并使用初始学习率 opt.lr_now 进行优化
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))
    # 过滤出验证模式
    if opt.is_load or opt.is_eval:
        print(opt.ckpt)
        # 已经训练的模型路径
        model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        # torch 加载模型
        ckpt = torch.load(model_path_len)
        # 从上个模型结束的地方开始验证
        start_epoch = ckpt['epoch'] + 1
        err_best = ckpt['err']
        # 获取该模型学习率
        lr_now = ckpt['lr']
        # 得到状态字典，包含了预训练模型的所有参数
        net_pred.load_state_dict(ckpt['state_dict'])
        # net.load_state_dict(ckpt)
        # optimizer.load_state_dict(ckpt['optimizer'])
        # lr_now = util.lr_decay_mine(optimizer, lr_now, 0.2)
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    print('>>> loading datasets')

    # 训练模式
    if not opt.is_eval:
        # 拉取数据集
        dataset = datasets.Datasets(opt, split=0)
        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        # DataLoader 封装
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=False)
        # 验证数据集
        valid_dataset = datasets.Datasets(opt, split=1)
        print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
        valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=0,
                                  pin_memory=False)

    # 测试数据集定义以及载入
    test_dataset = datasets.Datasets(opt, split=2)
    print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                             pin_memory=False)

    # evaluation
    if opt.is_eval:
        # 开始跑模型
        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt, epo=0)
        # 日志信息回显
        ret_log = np.array([])
        head = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, [k])
        log.save_csv_log(opt, head, ret_log, is_create=True, file_name='test_walking')
        # print('testing error: {:.3f}'.format(ret_test['m_p3d_h36']))
    # training
    if not opt.is_eval:
        # 轮数自定义,该参数是可调参数
        err_best = 1000
        # 迭代训练
        for epo in range(start_epoch, opt.epoch + 1):
            # 训练当中不一定是最好的结果
            is_best = False
            # if epo % opt.lr_decay == 0:
            # 学习率衰减用于拟合数据
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))
            print('>>> training epoch: {:d}'.format(epo))
            # 下面是跑训练/验证/测试三项模型
            ret_train = run_model(net_pred, optimizer, is_train=0, data_loader=data_loader, epo=epo, opt=opt)
            print('train error: {:.3f}'.format(ret_train['m_p3d_h36']))
            ret_valid = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt, epo=epo)
            print('validation error: {:.3f}'.format(ret_valid['m_p3d_h36']))
            ret_test = run_model(net_pred, is_train=2, data_loader=test_loader, opt=opt, epo=epo)
            print('testing error: {:.3f}'.format(ret_test['#1']))
            # 回显日志
            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            for k in ret_valid.keys():
                ret_log = np.append(ret_log, [ret_valid[k]])
                head = np.append(head, ['valid_' + k])
            for k in ret_test.keys():
                ret_log = np.append(ret_log, [ret_test[k]])
                head = np.append(head, ['test_' + k])
            # 测试文件CSV格式输出
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
            # 更新最好的模型
            if ret_valid['m_p3d_h36'] < err_best:
                err_best = ret_valid['m_p3d_h36']
                is_best = True
            # 各指标输出
            log.save_ckpt({'epoch': epo,
                           'lr': lr_now,
                           'err': ret_valid['m_p3d_h36'],
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          is_best=is_best, opt=opt)


def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None):
    # 选则模式：训练/评估
    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()
    # 日志value
    l_p3d = 0
    # l_beta = 0
    # j17to14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]
    if is_train <= 1:
        m_p3d_h36 = 0
    else:
        titles = np.array(range(opt.output_n)) + 1
        m_p3d_h36 = np.zeros([opt.output_n])
    # 各参数记录
    n = 0
    itera = 1
    in_n = opt.input_n
    out_n = opt.output_n
    joint_used = np.arange(4, 22)
    seq_in = opt.kernel_size
    idx = np.expand_dims(np.arange(seq_in + 1), axis=1) + np.expand_dims(np.arange(out_n), axis=0)
    # 耗时记录
    st = time.time()
    for i, (p3d_h36) in enumerate(data_loader):
        batch_size, seq_n, _, _ = p3d_h36.shape
        n += batch_size
        bt = time.time()
        p3d_h36 = p3d_h36.float().cuda()[:, :, joint_used] * 1000
        p3d_sup = p3d_h36.clone()[:, -out_n - seq_in:]
        p3d_src = p3d_h36.clone().reshape([batch_size, in_n + out_n, len(joint_used) * 3])
        # 跑预测
        p3d_out_all = net_pred(p3d_src, output_n=out_n, input_n=in_n, itera=itera)

        p3d_out = p3d_out_all[:, seq_in:].reshape([batch_size, out_n, len(joint_used), 3])

        p3d_out_all = p3d_out_all[:, :, 0].reshape([batch_size, seq_in + out_n, len(joint_used), 3])

        # training process
        # 2d joint loss:
        grad_norm = 0
        if is_train == 0:
            # loss_p3d = torch.mean(torch.sum(torch.abs(p3d_out_all - p3d_sup), dim=4))
            loss_p3d = torch.mean(torch.norm(p3d_out_all - p3d_sup, dim=3))

            loss_all = loss_p3d
            optimizer.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
            optimizer.step()
            # update log values
            l_p3d += loss_p3d.cpu().data.numpy() * batch_size

        if is_train <= 1:  # if is validation or train simply output the overall mean error
            # 归一化 -> 平均池化
            mpjpe_p3d_h36 = torch.mean(torch.norm(p3d_h36[:, in_n:in_n + out_n] - p3d_out, dim=3))
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size
        else:
            # 归一化 -> 平均池化 -> 求和
            mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()

        if i % 1000 == 0:
            print('{}/{}|bt {:.3f}s|tt{:.0f}s'.format(i + 1, len(data_loader), time.time() - bt, time.time() - st))

    ret = {}

    # 结果损失输出
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n

    if is_train <= 1:
        ret["m_p3d_h36"] = m_p3d_h36 / n

    else:
        m_p3d_h36 = m_p3d_h36 / n
        for j in range(out_n):
            ret["#{:d}".format(titles[j])] = m_p3d_h36[j]

    return ret


if __name__ == '__main__':
    option = Options().parse()
    main(option)
