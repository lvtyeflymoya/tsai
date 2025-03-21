from tsai.basics import *
import sklearn
import logging
import os 
from pathlib import Path
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error


# 创建实验目录
base_dir = Path("trainResult")
existing_exps = [d.name for d in base_dir.glob("experiment*") if d.is_dir()]
exp_numbers = [int(exp[10:]) for exp in existing_exps if exp[10:].isdigit()]
next_exp = max(exp_numbers) + 1 if exp_numbers else 1

exp_path = base_dir / f"experiment{next_exp}"
(exp_path / "metrics").mkdir(parents=True, exist_ok=True)
(exp_path / "model").mkdir(parents=True, exist_ok=True)

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # 输出到控制台
        # 可以添加更多的处理器，如将日志写入文件
        # logging.FileHandler('preprocessing.log')
    ]
)

# 加载数据集
dsid = "ETTh1"
df_raw = get_long_term_forecasting_data(dsid, target_dir="D:/Python_Project/toolscript", task='S')
# print(df_raw)

# 数据预处理
datetime_col = "date"
freq = 's'
columns = df_raw.columns[1:]
method = 'ffill'
value = 0

# pipeline
preproc_pipe = sklearn.pipeline.Pipeline([
    ('shrinker', TSShrinkDataFrame()), # shrink dataframe memory usage
    ('drop_duplicates', TSDropDuplicates(datetime_col=datetime_col)), # drop duplicate rows (if any)
    ('add_mts', TSAddMissingTimestamps(datetime_col=datetime_col, freq=freq)), # add missing timestamps (if any)
    ('fill_missing', TSFillMissing(columns=columns, method=method, value=value)), # fill missing data (1st ffill. 2nd value=0)
    ], 
    verbose=True)
mkdir('data', exist_ok=True, parents=True)
save_object(preproc_pipe, 'data/preproc_pipe.pkl') # 将预处理流水线对象保存为一个 pickle 文件
preproc_pipe = load_object('data/preproc_pipe.pkl')

df = preproc_pipe.fit_transform(df_raw)
# logging.info("预处理后的数据内容：")
# logging.info(df)

# 数据划分
fcst_history = 240 # steps in the past
fcst_horizon = 30  # steps in the future
valid_size   = 0.1  # int or float indicating the size of the validation set
test_size    = 0.2  # int or float indicating the size of the test set

splits = get_long_term_forecasting_splits(df, fcst_history=fcst_history, 
                                          fcst_horizon=fcst_horizon, dsid=dsid, show_plot=True)
# logging.info("分割后的数据内容：")
# logging.info(splits)

# 数据标准化，打分数据
columns = df.columns[1:]
train_split = splits[0]

exp_pipe = sklearn.pipeline.Pipeline([
    ('scaler', TSStandardScaler(columns=columns)), # standardize data using train_split
    ], 
    verbose=True)
save_object(exp_pipe, 'data/exp_pipe.pkl')
exp_pipe = load_object('data/exp_pipe.pkl')

df_scaled = exp_pipe.fit_transform(df, scaler__idxs=train_split)
logging.info("标准化后的数据内容：")
logging.info(df_scaled)

# 应用滑动窗口
x_vars = df.columns[1:]
y_vars = df.columns[1:]
X, y = prepare_forecasting_data(df, fcst_history=fcst_history, fcst_horizon=fcst_horizon, x_vars=x_vars, y_vars=y_vars)
logging.info("滑动窗口后的数据内容：")
logging.info(f"X.shape: {X.shape}, y.shape: {y.shape}")


# 准备预测器，可以理解为准备训练参数
arch_config = dict(
    n_layers=3,  # number of encoder layers
    n_heads=4,  # number of heads
    d_model=16,  # dimension of model
    d_ff=128,  # dimension of fully connected network
    attn_dropout=0.0, # dropout applied to the attention weights
    dropout=0.3,  # dropout applied to all linear layers in the encoder except q,k&v projections
    patch_len=16,  # length of the patch applied to the time series to create patches
    stride=8,  # stride used when creating patches
    padding_patch=True,  # padding_patch
)
learn = TSForecaster(X, y, splits=splits, batch_size=1024, path=str(exp_path), pipelines=[preproc_pipe, exp_pipe],
                     arch="PatchTST", arch_config=arch_config, metrics=[mse, mae])
learn.dls.valid.drop_last = True
logging.info(learn.summary())

# 训练模型
n_epochs = 10
lr_max = 0.0025
# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', type=str.lower, choices=['true', 'false'], default='false',
                    help='是否加载预训练模型（true/false）')
parser.add_argument('--pretrained_path', type=str, default='D:/Python_Project/tsai/trainResult/experiment11/model/PatchTST_best.pth',
                    help='预训练模型路径')
args = parser.parse_args()

weights_path = Path(args.pretrained_path)
# 在训练开始前加载预训练模型
if args.pretrained == 'false':
    if not Path(args.pretrained_path).exists():
        raise FileNotFoundError(f"预训练模型未找到：{args.pretrained_path}")
    learn = TSForecaster(X, y, splits=splits, batch_size=16, path=str(exp_path), pipelines=[preproc_pipe, exp_pipe],
                     arch="PatchTST", arch_config=arch_config, metrics=[mse, mae],
                     pretrained=True, weights_path=weights_path)
    # learn = TSForecaster(X, y, splits=splits, batch_size=16, path=str(exp_path), pipelines=[preproc_pipe, exp_pipe],
    #                  arch="PatchTST", arch_config=arch_config, metrics=[mse, mae])
    # learn.load(args.pretrained_path)
    logging.info(f"已加载预训练模型：{args.pretrained_path}")
    learn.freeze_to(-1)  # 冻结除最后一层外的所有层

# 在训练开始前初始化最佳指标
best_mse = float('inf')
best_mae = float('inf')
results_df = pd.DataFrame(columns=["mse", "mae"])
val_interval = 2

# 训练循环:每val_interval个epoch验证一次
for epoch_start in range(0, n_epochs, val_interval):

    learn.fit_one_cycle(val_interval, lr_max=lr_max)
    
    # 验证集预测
    scaled_preds, *_ = learn.get_X_preds(X[splits[1]])
    scaled_preds = to_np(scaled_preds)
    scaled_y_true = y[splits[1]]
    
    # 计算当前指标
    current_mse = mean_squared_error(scaled_y_true.flatten(), scaled_preds.flatten())
    current_mae = mean_absolute_error(scaled_y_true.flatten(), scaled_preds.flatten())
    
    # 记录结果
    results_df.loc[f"epoch_{epoch_start + val_interval}"] = [current_mse, current_mae]
    
    # 保存最佳模型
    if current_mse < best_mse and current_mae < best_mae:
        best_mse = current_mse
        best_mae = current_mae
        torch.save(learn.model.state_dict(), exp_path / "model/PatchTST_best.pth")
        logging.info(f"Epoch {epoch_start + val_interval}: 模型已保存，当前最佳MSE: {best_mse:.4f}, 当前最佳MAE: {best_mae:.4f}")

# 保存最终模型和验证结果

torch.save(learn.model.state_dict(), exp_path / "model/patchTST.pth")
logging.info("训练完成，最终模型和验证结果已保存")
learn.plot_metrics()

plt.savefig(str(exp_path / "metrics/training_metrics.png"))
plt.close()