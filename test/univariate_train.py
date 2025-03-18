import torch
from tsai.models.PatchTST import PatchTST
from tsai.basics import SlidingWindow, TimeSplitter, TSForecasting

# 单变量数据参数
bs = 32
seq_len = 60       # 输入序列长度
pred_dim = 1       # 预测下一个时间步
c_in = 1           # 单变量输入通道
c_out = 1          # 单变量输出通道

# 生成示例数据（替换为您的真实数据）
# 生成示例数据后转换为PyTorch张量
data = torch.randn(1000)  # 假设有1000个时间点的数据
X, y = SlidingWindow(window_len=seq_len, horizon=pred_dim)(data)
X = torch.from_numpy(X).float()  # 转换为PyTorch张量
y = torch.from_numpy(y).float()

# 划分训练验证集
splits = TimeSplitter(800)(y.numpy())  # 使用numpy数组进行划分

# 模型配置
arch_config = dict(
    n_layers=3,
    n_heads=8,
    d_model=64,
    d_ff=128,
    attn_dropout=0.1,
    dropout=0.2,
    patch_len=12,
    stride=6,
    individual=False,
    padding_patch=False  # 添加此参数禁用自动填充
)

# 创建模型
model = PatchTST(c_in, c_out, seq_len, pred_dim, **arch_config)

# 训练配置
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环
n_epochs = 50
# 修改训练循环中的数据处理部分
for epoch in range(n_epochs):
    # 随机选择批次
    idx = torch.randint(0, len(splits[0]), (bs,))
    xb = X[splits[0][idx]].unsqueeze(1)  # 添加通道维度 [bs, 1, seq_len]
    yb = y[splits[0][idx]].unsqueeze(1)  # [bs, 1, pred_dim]
    
    # 前向传播
    outputs = model(xb)
    loss = criterion(outputs, yb)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 打印训练信息
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

# 保存模型
torch.save(model.state_dict(), 'models/univariate_PatchTST.pth')
