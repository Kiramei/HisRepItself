# import torch
# import torch.nn as nn
# import numpy as np

# from model import FFC


# def data_loader() -> torch.Tensor:
#     return torch.rand(size=(32, 66, 50))


# class TestUnit(nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(in_features=50, out_features=256, bias=False),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(in_features=256, out_features=128, bias=False),
#             nn.ReLU(),
#             nn.Linear(in_features=128, out_features=50, bias=False),
#             nn.Softmax(dim=(-1))
#         )
#         # self.ffc = FFC.FFC(in_channels=50, out_channels=50, kernel_size=10, ratio_gin=0.5, ratio_gout=0.5)
#         self.ffc = FFC.FourierUnit(in_channels=50, out_channels=50, groups=1)

#     def forward(self, x):
#         x = self.mlp(x)
#         x = self.ffc(x)
#         x = self.mlp(x)
#         return x


# if __name__ == '__main__':
#     data_set = data_loader()
#     validate = data_loader()
#     net = TestUnit()
#     for epoch in range(20):
#         res = net(data_set)
#         loss = torch.mean(torch.norm(res - validate, dim=2))

# original_list = [1, 2, 3, 4]
# new_list = [num for num in original_list for _ in range(10)]

# print(new_list)

# # new_list = [num for (i,num) in enumerate(new_list) if i % 10 == 0]
# average = [sum(new_list[i:i+10])//10 for i in range(0, len(new_list), 10)]
# print(average)

import numpy as np

# original_array = np.array([[1, 2, 3, 4]])  # 原始数组，形状为 (1, 4)
# new_array = np.repeat(original_array, 80, axis=1)  # 将原始数组在第二维度上重复 80 次

# new_shape = new_array.shape  # 新数组的形状

# print(new_array.shape)


src_data = np.random.rand(1, 1, 2)

data_tmp = np.repeat(src_data, 10, axis=2)

print(data_tmp)