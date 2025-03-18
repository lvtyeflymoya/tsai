import torch
from tsai.models.PatchTST import PatchTST

# 定义数据参数
c_in = 9  # 输入通道数
c_out = 1
seq_len = 60
pred_dim = 20

# 模型配置，需要和训练时的配置保持一致
arch_config = dict(
    n_layers=3,
    n_heads=16,
    d_model=128,
    d_ff=256,
    attn_dropout=0.,
    dropout=0.2,
    patch_len=16,
    stride=8,
)

# 创建模型实例
model = PatchTST(c_in, c_out, seq_len, pred_dim, **arch_config)

# 加载训练好的模型参数
model.load_state_dict(torch.load('PatchTST_model.pth'))
model.eval()  # 设置模型为评估模式

# 准备输入数据，这里使用随机数据作为示例，实际应用中应替换为真实数据
bs = 32  # 批量大小
input_data = torch.randn(bs, c_in, seq_len)

# 进行推理预测
with torch.no_grad():  # 不计算梯度，节省内存
    predictions = model(input_data)

# 打印预测结果
print("预测结果的形状:", predictions.shape)
print("预测结果:", predictions)