import torch
from tsai.models.PatchTST import PatchTST

# 加载训练好的模型
model = PatchTST(c_in=1, c_out=1, seq_len=60, pred_dim=1, 
                n_layers=3, n_heads=8, d_model=64, d_ff=128,
                patch_len=12, stride=6)
model.load_state_dict(torch.load('univariate_PatchTST.pth'))
model.eval()

# 使用最新60个时间步进行预测
input_sequence = torch.randn(60)  # 替换为真实数据
with torch.no_grad():
    prediction = model(input_sequence.unsqueeze(0).unsqueeze(0))  # 添加批次和通道维度
    print(f'下一个时间步预测值: {prediction.item():.4f}')
