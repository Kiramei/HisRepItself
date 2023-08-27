from torch.nn import Module
from torch import nn
import torch
# import model.transformer_base
# import math
from model import GCN
import utils.util as util
import numpy as np

'''
注意力模型：Dot-Product -> Normalization -> Weight & Sum -> Output
          q ↑  j_i ↑                        V_i ↑
'''


class AttModel(Module):

    def __init__(self, in_features=48, kernel_size=5, d_model=512, num_stage=2, dct_n=10, dropout=0.3):
        super(AttModel, self).__init__()
        # 卷积核与模型
        self.kernel_size = kernel_size
        self.d_model = d_model
        # self.seq_in = seq_in
        # 字典大小
        self.dct_n = dct_n
        # ks = int((kernel_size + 1) / 2)
        # 预训练模型的卷积呵结果
        assert kernel_size == 10
        # Q 和 k 网络均这样构造: 1DConv(6 ks) -> ReLU -> 1DConv(5 ks) -> ReLU
        # 由两个卷积层和两个ReLU激活函数按顺序连接而成。第一个卷积层将输入通道数为3的图像
        # 数据转换为16个输出通道，第二个卷积层将16个输入通道转换为32个输出通道。每个卷积层
        # 之后都有一个ReLU激活函数对输出进行非线性变换。
        # => Essay Page No.8
        # q 卷积序列
        self.convQ = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())
        # k卷积序列
        self.convK = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())
        # 图神经网络初始化
        self.gcn = GCN.GCN(input_feature=dct_n, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)

        self.mlp = nn.Sequential(nn.Linear(in_features=(dct_n * 2), out_features=d_model*2),
                                 nn.ReLU(),
                                 nn.Linear(in_features=d_model*2, out_features=d_model),
                                 nn.ReLU(),
                                 nn.Linear(in_features=d_model, out_features=(dct_n * 2)))

    def forward(self, src, output_n=25, input_n=50, itera=1):
        """

        :param src: [batch_size,seq_len,feat_dim]
        :param output_n:
        :param input_n:
        :param itera:
        :return:
        """
        # 拉取字典矩阵大小/源矩阵(=> key 矩阵 + query 矩阵)/批量大小,此部分相当于decode
        # output_n: 10
        dct_n = self.dct_n  # 20
        src = src[:, :input_n]  # [bs,in_n,dim] -> [32,50,66]
        src_tmp = src.clone()  # [32,50,66]
        bs = src.shape[0]  # batch_size: 32
        src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()  # [32,66,40]
        src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()  # [32,66,10]
        # 获取DCT变化后的矩阵以及其逆矩阵,转移cuda
        dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)  # [20,20] [20,20]
        dct_m = torch.from_numpy(dct_m).float().cuda()  # [20,20] (dct_n, dct_n)
        idct_m = torch.from_numpy(idct_m).float().cuda()  # [20,20] (dct_n, dct_n)

        vn = input_n - self.kernel_size - output_n + 1  # 50 - 10 - 10 + 1 = 31
        vl = self.kernel_size + output_n  # 10 + 10 = 20
        # idx : (31,20)
        idx = np.expand_dims(np.arange(vl), axis=0) + np.expand_dims(np.arange(vn), axis=1)
        # [32,50,66] => [32,(31,20),66] => [32*31,20,66]
        src_value_tmp = src_tmp[:, idx].clone().reshape(
            [bs * vn, vl, -1])
        # [32*31,20,66] * [1,20,20] = [32*31,20,66] => [32,31,20,66] => [32,31,66*20]
        src_value_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp) \
            .reshape([bs, vn, dct_n, -1]) \
            .transpose(2, 3).reshape(
            [bs, vn, -1])  # [32,40,66*11] 错误的维度，应该是参数不同的原因
        # 得到index，便于迭代中抽取维度，为GCN服务
        # 最后一帧拓展为10帧
        # [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n
        outputs = []
        # 利用K的卷积神经网络对源K操作，K只有一项
        # [32,66,40] => [32,256,31]
        key_tmp = self.convK(src_key_tmp / 1000.0)
        for i in range(itera):
            # 量化询问,并作卷积化
            # [32,66,10] => [32,256,1]
            query_tmp = self.convQ(src_query_tmp / 1000.0)
            # Attention Model基本要素：得分
            # [32,1,31]
            score_tmp = torch.matmul(query_tmp.transpose(1, 2), key_tmp) + 1e-15
            # 量化Score,得到注意力
            # [32,1,31] / [32,1,1] = [32,1,31]
            att_tmp = score_tmp / (torch.sum(score_tmp, dim=2, keepdim=True))
            # 注意力回乘,得到注意力
            # [32,66,20]
            dct_att_tmp = torch.matmul(att_tmp, src_value_tmp)[:, 0].reshape(
                [bs, -1, dct_n])
            # 抽取向量提供GCN的Input
            # [32,20,66]，抽取最后一帧，拓展为11帧，为query
            input_query = src_tmp[:, idx]
            '''
            代码使用 DCT 矩阵 dct_m 对输入张量 input_gcn 进行变换，并将其形状变为 (bs, vn, dct_n)。
            其中 dct_n 表示 DCT 变换后的特征维度数量，vn 表示需要用到的时间步数量。
            然后，代码将 dct_att_tmp 拼接到 dct_in_tmp 中的最后一维，得到形状为 (bs, vn, dct_n+att_size) 的张量 dct_in_tmp，
            其中 att_size 表示注意力向量的长度。
            '''
            # [32,20,66] * [1,20,20] => [32,20,66] => [32,66,20]
            gcn_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_query).transpose(1, 2)
            # => [32,66,20+20] => [32,66,40]
            # 这里作gcn
            gcn_out_tmp = self.gcn(gcn_in_tmp)
            # 在mlp层前拼接GCN输出与注意力
            mlp_in_tmp = torch.cat([gcn_out_tmp, dct_att_tmp], dim=-1)
            # 加入图神经网路计算
            # => [32,66,40]
            # dct_out_tmp = self.gcn(dct_in_tmp)

            # 这里换成MLP，不改变维度
            mlp_out_tmp = self.mlp(mlp_in_tmp)

            # 代码将 dct_out_tmp 中的前 dct_n 个特征维度取出，并通过 IDCT 矩阵 idct_m 进行变换，
            # 得到形状为 (bs, vn, output_n) 的张量 out_gcn。其中 output_n 表示 GCN 操作的输出维度数量。
            # [32,66,20] * [1,20,20] => [32,66,20] => [32,20,66]
            out_mlp = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                                   mlp_out_tmp[:, :, :dct_n].transpose(1, 2))

            # 添加输出 [32,20,66] => [32,20,1,66]
            outputs.append(out_mlp.unsqueeze(2))

            # 在迭代次数大于1时,将Key-value矩阵更新,将操作后的值还原回矩阵当中,相当于重新encode
            if itera > 1:
                # 取最后output_n帧
                out_tmp = out_mlp.clone()[:, 0 - output_n:]
                # 将原src与输出帧拼接
                src_tmp = torch.cat([src_tmp, out_tmp], dim=1)
                # 算个数
                vn = 1 - 2 * self.kernel_size - output_n
                vl = self.kernel_size + output_n
                idx_dct = np.expand_dims(np.arange(vl), axis=0) + \
                          np.expand_dims(np.arange(vn, -self.kernel_size - output_n + 1), axis=1)

                src_key_tmp = src_tmp[:, idx_dct[0, :-1]].transpose(1, 2)
                key_new = self.convK(src_key_tmp / 1000.0)
                key_tmp = torch.cat([key_tmp, key_new], dim=2)

                src_dct_tmp = src_tmp[:, idx_dct].clone().reshape(
                    [bs * self.kernel_size, vl, -1])
                src_dct_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_dct_tmp).reshape(
                    [bs, self.kernel_size, dct_n, -1]).transpose(2, 3).reshape(
                    [bs, self.kernel_size, -1])
                src_value_tmp = torch.cat([src_value_tmp, src_dct_tmp], dim=1)

                src_query_tmp = src_tmp[:, -self.kernel_size:].transpose(1, 2)
        # [32,20,1,66] => [32,20,66]
        outputs = torch.cat(outputs, dim=2)
        # outputs = src_tmp[:, -35:, :].unsqueeze(2)

        return outputs


class AttModelPerParts(Module):

    def __init__(self, in_features=48, kernel_size=5, d_model=512, num_stage=2, dct_n=10,
                 parts_idx=None):
        super(AttModelPerParts, self).__init__()

        if parts_idx is None:
            parts_idx = [[1, 2, 3], [4, 5, 6]]
        self.kernel_size = kernel_size
        self.d_model = d_model
        self.parts_idx = parts_idx
        self.in_features = in_features
        self.dct_n = dct_n
        # ks = int((kernel_size + 1) / 2)
        assert kernel_size == 10
        convQ = []
        convK = []
        # 在 part 部分K和Q呈现序列形式
        for i in range(len(parts_idx)):
            pi = parts_idx[i]
            convQ.append(nn.Sequential(nn.Conv1d(in_channels=len(pi), out_channels=d_model, kernel_size=6,
                                                 bias=False),
                                       nn.ReLU(),
                                       nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                                 bias=False),
                                       nn.ReLU()))
            convK.append(nn.Sequential(nn.Conv1d(in_channels=len(pi), out_channels=d_model, kernel_size=6,
                                                 bias=False),
                                       nn.ReLU(),
                                       nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                                 bias=False),
                                       nn.ReLU()))
        self.convQ = nn.ModuleList(convQ)
        self.convK = nn.ModuleList(convK)

        self.gcn = GCN.GCN(input_feature=dct_n * 2, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)

        # self.mlp = torch.nn.

    def forward(self, src, output_n=25, input_n=50, itera=1):
        """

        :param src: [batch_size,seq_len,feat_dim]
        :param output_n:
        :param input_n:
        :param itera:
        :return:
        """
        dct_n = self.dct_n
        src = src[:, :input_n]  # [bs,in_n,dim]
        src_tmp = src.clone()
        bs = src.shape[0]
        src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()
        src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()

        dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)
        dct_m = torch.from_numpy(dct_m).float().cuda()
        idct_m = torch.from_numpy(idct_m).float().cuda()

        vn = input_n - self.kernel_size - output_n + 1
        vl = self.kernel_size + output_n
        idx = np.expand_dims(np.arange(vl), axis=0) + \
              np.expand_dims(np.arange(vn), axis=1)
        src_value_tmp = src_tmp[:, idx].clone().reshape(
            [bs * vn, vl, -1])
        src_value_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp).reshape(
            [bs, vn, dct_n, -1]).transpose(2, 3)  # .reshape([bs, vn, -1])  # [32,40,66*11]

        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n
        outputs = []
        key_tmp = []
        for ii, pidx in enumerate(self.parts_idx):
            key_tmp.append(self.convK[ii](src_key_tmp[:, pidx] / 1000.0).unsqueeze(1))
        key_tmp = torch.cat(key_tmp, dim=1)

        for i in range(itera):
            query_tmp = []
            for ii, pidx in enumerate(self.parts_idx):
                query_tmp.append(self.convQ[ii](src_query_tmp[:, pidx] / 1000.0).unsqueeze(1))
            query_tmp = torch.cat(query_tmp, dim=1)
            score_tmp = torch.matmul(query_tmp.transpose(2, 3), key_tmp) + 1e-15
            att_tmp = score_tmp / (torch.sum(score_tmp, dim=3, keepdim=True))
            dct_att_tmp = torch.zeros([bs, self.in_features, self.dct_n]).float().cuda()
            for ii, pidx in enumerate(self.parts_idx):
                dct_att_tt = torch.matmul(att_tmp[:, ii], src_value_tmp[:, :, pidx]
                                          .reshape([bs, -1, len(pidx) * self.dct_n])).squeeze(1)
                dct_att_tt = dct_att_tt.reshape([bs, len(pidx), self.dct_n])
                dct_att_tmp[:, pidx, :] = dct_att_tt

            input_gcn = src_tmp[:, idx]
            dct_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)
            dct_in_tmp = torch.cat([dct_in_tmp, dct_att_tmp], dim=-1)
            dct_out_tmp = self.gcn(dct_in_tmp)
            out_gcn = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                                   dct_out_tmp[:, :, :dct_n].transpose(1, 2))
            outputs.append(out_gcn.unsqueeze(2))
            # update key-value query
            out_tmp = out_gcn.clone()[:, 0 - output_n:]
            src_tmp = torch.cat([src_tmp, out_tmp], dim=1)

            vn = 1 - 2 * self.kernel_size - output_n
            vl = self.kernel_size + output_n
            idx_dct = np.expand_dims(np.arange(vl), axis=0) + \
                      np.expand_dims(np.arange(vn, -self.kernel_size - output_n + 1), axis=1)

            src_key_tmp = src_tmp[:, idx_dct[0, :-1]].transpose(1, 2)

            key_new = []
            for ii, pidx in enumerate(self.parts_idx):
                key_new.append(self.convK[ii](src_key_tmp[:, pidx] / 1000.0).unsqueeze(1))
            key_new = torch.cat(key_new, dim=1)
            key_tmp = torch.cat([key_tmp, key_new], dim=3)

            src_dct_tmp = src_tmp[:, idx_dct].clone().reshape(
                [bs * self.kernel_size, vl, -1])
            src_dct_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_dct_tmp).reshape(
                [bs, self.kernel_size, dct_n, -1]).transpose(2, 3)  # .reshape([bs, self.kernel_size, -1])
            src_value_tmp = torch.cat([src_value_tmp, src_dct_tmp], dim=1)

            src_query_tmp = src_tmp[:, -self.kernel_size:].transpose(1, 2)

        outputs = torch.cat(outputs, dim=2)
        return outputs
