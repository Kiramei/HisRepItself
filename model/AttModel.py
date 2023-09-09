from torch.nn import Module
from torch import nn
import torch
# import model.transformer_base
# import math
from model import GCN
from model.FFC import *
import utils.util as util
import numpy as np

'''
×¢ÒâÁ¦Ä£ÐÍ£ºDot-Product -> Normalization -> Weight & Sum -> Output
          q ¡ü  j_i ¡ü                        V_i ¡ü
'''


class AttModel(Module):

    def __init__(self, in_features=48, kernel_size=5, d_model=512, num_stage=2, dct_n=10, dropout=0.1):
        super(AttModel, self).__init__()
        # ¾í»ýºËÓëÄ£ÐÍ
        self.kernel_size = kernel_size
        self.d_model = d_model
        # self.seq_in = seq_in
        # ×Öµä´óÐ¡
        self.dct_n = dct_n
        # ks = int((kernel_size + 1) / 2)
        # Ô¤ÑµÁ·Ä£ÐÍµÄ¾í»ýºÇ½á¹û
        assert kernel_size == 10
        # Q ºÍ k ÍøÂç¾ùÕâÑù¹¹Ôì: 1DConv(6 ks) -> ReLU -> 1DConv(5 ks) -> ReLU
        # ÓÉÁ½¸ö¾í»ý²ãºÍÁ½¸öReLU¼¤»îº¯Êý°´Ë³ÐòÁ¬½Ó¶ø³É¡£µÚÒ»¸ö¾í»ý²ã½«ÊäÈëÍ¨µÀÊýÎª3µÄÍ¼Ïñ
        # Êý¾Ý×ª»»Îª16¸öÊä³öÍ¨µÀ£¬µÚ¶þ¸ö¾í»ý²ã½«16¸öÊäÈëÍ¨µÀ×ª»»Îª32¸öÊä³öÍ¨µÀ¡£Ã¿¸ö¾í»ý²ã
        # Ö®ºó¶¼ÓÐÒ»¸öReLU¼¤»îº¯Êý¶ÔÊä³ö½øÐÐ·ÇÏßÐÔ±ä»»¡£
        # => Essay Page No.8
        # q ¾í»ýÐòÁÐ
        self.convQ = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())
        # k¾í»ýÐòÁÐ
        self.convK = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())
        # Í¼Éñ¾­ÍøÂç³õÊ¼»¯
        self.gcn = GCN.GCN(input_feature=dct_n, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)

        self.mlp = nn.Sequential(nn.Linear(in_features=(dct_n * 2), out_features=(d_model * 2)),
                                 nn.ReLU(),
                                 nn.Linear(in_features=(d_model * 2), out_features=(dct_n * 2)),
                                 )

        # self.ffc = FFC.FFC(in_channels=in_features, out_channels=in_features, kernel_size=10, ratio_gin=0, ratio_gout=0)
        self.ffc = nn.Sequential(ResnetBlock_remove_IN(dim=in_features))
        self.linear = nn.Linear(in_features=400, out_features=20)
        # self.fft = nn.Sequential(nn.)

    def forward(self, src, output_n=25, input_n=50, itera=1):
        """

        :param src: [batch_size,seq_len,feat_dim]
        :param output_n:
        :param input_n:
        :param itera:
        :return:
        """
        # À­È¡×Öµä¾ØÕó´óÐ¡/Ô´¾ØÕó(=> key ¾ØÕó + query ¾ØÕó)/ÅúÁ¿´óÐ¡,´Ë²¿·ÖÏàµ±ÓÚdecode
        # output_n: 10
        dct_n = self.dct_n  # 20
        src = src[:, :input_n]  # [bs,in_n,dim] -> [32,50,66]
        src_tmp = src.clone()  # [32,50,66]
        bs = src.shape[0]  # batch_size: 32
        src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()  # [32,66,40]
        src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()  # [32,66,10]
        # »ñÈ¡DCT±ä»¯ºóµÄ¾ØÕóÒÔ¼°ÆäÄæ¾ØÕó,×ªÒÆcuda
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
            [bs, vn, -1])  # [32,40,66*11] ´íÎóµÄÎ¬¶È£¬Ó¦¸ÃÊÇ²ÎÊý²»Í¬µÄÔ­Òò
        # µÃµ½index£¬±ãÓÚµü´úÖÐ³éÈ¡Î¬¶È£¬ÎªGCN·þÎñ
        # ×îºóÒ»Ö¡ÍØÕ¹Îª10Ö¡
        # [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n
        outputs = []
        # ÀûÓÃKµÄ¾í»ýÉñ¾­ÍøÂç¶ÔÔ´K²Ù×÷£¬KÖ»ÓÐÒ»Ïî
        # [32,66,40] => [32,256,31]
        key_tmp = self.convK(src_key_tmp / 1000.0)
        for i in range(itera):
            # Á¿»¯Ñ¯ÎÊ,²¢×÷¾í»ý»¯
            # [32,66,10] => [32,256,1]
            query_tmp = self.convQ(src_query_tmp / 1000.0)
            # Attention Model»ù±¾ÒªËØ£ºµÃ·Ö
            # [32,1,31]
            score_tmp = torch.matmul(query_tmp.transpose(1, 2), key_tmp) + 1e-15
            # Á¿»¯Score,µÃµ½×¢ÒâÁ¦
            # [32,1,31] / [32,1,1] = [32,1,31]
            att_tmp = score_tmp / (torch.sum(score_tmp, dim=2, keepdim=True))
            # ×¢ÒâÁ¦»Ø³Ë,µÃµ½×¢ÒâÁ¦
            # [32,66,20]
            dct_att_tmp = torch.matmul(att_tmp, src_value_tmp)[:, 0].reshape(
                [bs, -1, dct_n])
            # ³éÈ¡ÏòÁ¿Ìá¹©GCNµÄInput
            # [32,20,66]£¬³éÈ¡×îºóÒ»Ö¡£¬ÍØÕ¹Îª11Ö¡£¬Îªquery
            input_query = src_tmp[:, idx]
            '''
            ´úÂëÊ¹ÓÃ DCT ¾ØÕó dct_m ¶ÔÊäÈëÕÅÁ¿ input_gcn ½øÐÐ±ä»»£¬²¢½«ÆäÐÎ×´±äÎª (bs, vn, dct_n)¡£
            ÆäÖÐ dct_n ±íÊ¾ DCT ±ä»»ºóµÄÌØÕ÷Î¬¶ÈÊýÁ¿£¬vn ±íÊ¾ÐèÒªÓÃµ½µÄÊ±¼ä²½ÊýÁ¿¡£
            È»ºó£¬´úÂë½« dct_att_tmp Æ´½Óµ½ dct_in_tmp ÖÐµÄ×îºóÒ»Î¬£¬µÃµ½ÐÎ×´Îª (bs, vn, dct_n+att_size) µÄÕÅÁ¿ dct_in_tmp£¬
            ÆäÖÐ att_size ±íÊ¾×¢ÒâÁ¦ÏòÁ¿µÄ³¤¶È¡£
            '''
            # [32,20,66] * [1,20,20] => [32,20,66] => [32,66,20]
            gcn_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_query).transpose(1, 2)
            # # => [32,66,20+20] => [32,66,40]
            # # ÕâÀï×÷gcn
            # gcn_out_tmp = torch.zeros((32, 66, 20))
            # gcn_in_tmp = [num for num in gcn_in_tmp for _ in range(20)]
            B, _, _ = gcn_in_tmp.shape
            gcn_out_tmp = gcn_in_tmp.clone().repeat(1, 1, 20)
            gcn_out_tmp = gcn_out_tmp.reshape(B, 66, 20, 20)
            gcn_out_tmp = self.ffc(gcn_out_tmp)
            # print(gcn_out_tmp.shape)
            gcn_out_tmp = self.linear(gcn_out_tmp.reshape(B, 66, 400))
            # gcn_out_tmp = [sum(new_list[i:i+10])//10 for i in range(0, len(gcn_out_tmp), 10)]
            mlp_in_tmp = torch.cat([gcn_out_tmp, dct_att_tmp], dim=-1)
            # ¼ÓÈëÍ¼Éñ¾­ÍøÂ·¼ÆËã
            # => [32,66,40]
            # ¼ÓÈëÍ¼Éñ¾­ÍøÂ·¼ÆËã
            # => [32,66,40]
            # ¼ÓÈëÍ¼Éñ¾­ÍøÂ·¼ÆËã
            # => [32,66,40]
            # ¼ÓÈëÍ¼Éñ¾­ÍøÂ·¼ÆËã
            # => [32,66,40]
            # ¼ÓÈëÍ¼Éñ¾­ÍøÂ·¼ÆËã
            # => [32,66,40]
            # dct_out_tmp = self.gcn(dct_in_tmp)

            # ÕâÀï»»³ÉMLP£¬²»¸Ä±äÎ¬¶È
            mlp_out_tmp = self.mlp(mlp_in_tmp)

            # ´úÂë½« dct_out_tmp ÖÐµÄÇ° dct_n ¸öÌØÕ÷Î¬¶ÈÈ¡³ö£¬²¢Í¨¹ý IDCT ¾ØÕó idct_m ½øÐÐ±ä»»£¬
            # µÃµ½ÐÎ×´Îª (bs, vn, output_n) µÄÕÅÁ¿ out_gcn¡£ÆäÖÐ output_n ±íÊ¾ GCN ²Ù×÷µÄÊä³öÎ¬¶ÈÊýÁ¿¡£
            # [32,66,20] * [1,20,20] => [32,66,20] => [32,20,66]
            out_mlp = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                                   mlp_out_tmp[:, :, :dct_n].transpose(1, 2))

            # Ìí¼ÓÊä³ö [32,20,66] => [32,20,1,66]
            outputs.append(out_mlp.unsqueeze(2))

            # ÔÚµü´ú´ÎÊý´óÓÚ1Ê±,½«Key-value¾ØÕó¸üÐÂ,½«²Ù×÷ºóµÄÖµ»¹Ô­»Ø¾ØÕóµ±ÖÐ,Ïàµ±ÓÚÖØÐÂencode
            if itera > 1:
                # È¡×îºóoutput_nÖ¡
                out_tmp = out_mlp.clone()[:, 0 - output_n:]
                # ½«Ô­srcÓëÊä³öÖ¡Æ´½Ó
                src_tmp = torch.cat([src_tmp, out_tmp], dim=1)
                # Ëã¸öÊý
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
        # ÔÚ part ²¿·ÖKºÍQ³ÊÏÖÐòÁÐÐÎÊ½
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


class ResnetBlock_remove_IN(nn.Module):
    def __init__(self, dim, dilation=1):
        super(ResnetBlock_remove_IN, self).__init__()

        self.ffc1 = FFC_BN_ACT(dim, dim, 3, 0.75, 0.75, stride=1, padding=1, dilation=dilation, groups=1, bias=False,
                               norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, enable_lfu=False)

        self.ffc2 = FFC_BN_ACT(dim, dim, 3, 0.75, 0.75, stride=1, padding=1, dilation=1, groups=1, bias=False,
                               norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, enable_lfu=False)

    def forward(self, x):
        output = x
        _, c, _, _ = output.shape
        output = torch.split(output, [c - int(c * 0.75), int(c * 0.75)], dim=1)
        x_l, x_g = self.ffc1(output)
        output = self.ffc2((x_l, x_g))
        output = torch.cat(output, dim=1)
        output = x + output

        return output
