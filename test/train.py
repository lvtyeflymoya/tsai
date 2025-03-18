import torch
from tsai.models.PatchTST import PatchTST
from fastcore.test import test_eq
from tsai.models.utils import count_parameters
from tsai.basics import *

# 定义数据参数
bs = 32
c_in = 9  # 输入通道数
c_out = 1
seq_len = 60
pred_dim = 20

# 生成随机输入数据
xb = torch.randn(bs, c_in, seq_len)

# 定义模型配置
arch_config = dict(
    n_layers=3,  # 编码器层数
    n_heads=16,  # 头的数量
    d_model=128,  # 模型维度
    d_ff=256,  # 全连接网络维度
    attn_dropout=0.,
    dropout=0.2,  # 编码器中所有线性层的 dropout
    patch_len=16,  # 补丁长度
    stride=8,  # 步长
)

# 创建模型
model = PatchTST(c_in, c_out, seq_len, pred_dim, **arch_config)

# 定义损失函数和优化器
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环
n_epochs = 10
for epoch in range(n_epochs):
    # 前向传播
    outputs = model(xb)
    loss = criterion(outputs, torch.randn(bs, c_in, pred_dim))

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印训练信息
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')

# 保存模型
torch.save(model.state_dict(), 'PatchTST_model.pth')