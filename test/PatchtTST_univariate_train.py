from tsai.basics import *
import sklearn
import logging
import os 
from pathlib import Path
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error


# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', type=str.lower, choices=['true', 'false'], default='false',
                    help='是否加载预训练模型（true/false）')
parser.add_argument('--pretrained_path', type=str, default='D:/Python_Project/tsai/trainResult/experiment11/model/PatchTST_best.pth',
                    help='预训练模型路径')
parser.add_argument('--dsid', type=str, default='ETTh1',help='数据集名称')
args = parser.parse_args()


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
dsid = args.dsid
df_raw = get_long_term_forecasting_data(dsid, target_dir="D:/Python_Project/toolscript/csvfile/down_inside", task='S')
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
    # ('add_mts', TSAddMissingTimestamps(datetime_col=datetime_col, freq=freq)), # add missing timestamps (if any)
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
fcst_history = 104 # steps in the past
fcst_horizon = 60  # steps in the future
valid_size   = 0.1  # int or float indicating the size of the validation set
test_size    = 0.2  # int or float indicating the size of the test set

splits = get_long_term_forecasting_splits(df, fcst_history=fcst_history, 
                                          fcst_horizon=fcst_horizon, dsid=dsid, show_plot=False)
# logging.info("分割后的数据内容：")
# logging.info(splits)

# 数据标准化，打分数据
columns = df.columns[1:]
train_split = splits[0]

exp_pipe = sklearn.pipeline.Pipeline([
    ('scaler', TSStandardScaler(columns=columns))   # standardize data using train_split
    ], 
    verbose=True)
save_object(exp_pipe, 'data/exp_pipe.pkl')
exp_pipe = load_object('data/exp_pipe.pkl')

df_scaled = exp_pipe.fit_transform(df, scaler__idxs=train_split)    # 确保只使用训练集进行标准化，之后将用于预测和验证集
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
n_epochs = 30
lr_max = 0.0025


# 在训练开始前加载预训练模型
if args.pretrained == 'true':
    if not Path(args.pretrained_path).exists():
        raise FileNotFoundError(f"预训练模型未找到：{args.pretrained_path}")
    learn = TSForecaster(X, y, splits=splits, batch_size=16, path=str(exp_path), pipelines=[preproc_pipe, exp_pipe],
                     arch="PatchTST", arch_config=arch_config, metrics=[mse, mae],
                     pretrained=True, weights_path=args.pretrained_path)
   
    logging.info(f"已加载预训练模型：{args.pretrained_path}")
    learn.freeze_to(-1)  # 冻结除最后一层外的所有层

# 在训练开始前初始化最佳指标
best_mse = float('inf')
best_mae = float('inf')
results_df = pd.DataFrame(columns=["mse", "mae"])
val_interval = 5
metrics_history = []    # 创建指标历史记录
no_improve_epochs = 0  # 早停计数器

# 训练循环:每val_interval个epoch验证一次
for epoch_start in range(0, n_epochs, val_interval):
    logging.info(f"Starting training from epoch {epoch_start + 1} to {epoch_start + val_interval}")
    
    # 指标记录
    before = len(learn.recorder.values)
    learn.fit_one_cycle(val_interval, lr_max=lr_max)
    # 获取本周期指标
    for i, record in enumerate(learn.recorder.values[before - 1:]):
        # epoch = epoch_start + i + 1
        epoch = (epoch_start // val_interval) * val_interval + i + 1
    
        train_loss = record[0]
        valid_loss = record[1]
        mse = record[2]
        mae = record[3] if len(record) > 3 else 0
        
        metrics_history.append((
            epoch, 
            float(train_loss),  # 直接转换为float
            float(valid_loss),
            float(mse),
            float(mae)
        ))
    
    # 记录结果
    results_df.loc[f"epoch_{epoch_start + val_interval}"] = [mse, mae]
    
    # 保存最佳模型
    if mse < best_mse and mae < best_mae:
        best_mse = mse
        best_mae = mae
        torch.save(learn.model.state_dict(), exp_path / "model/PatchTST_best.pth")
        logging.info(f"Epoch {epoch_start + val_interval}: 模型已保存，当前最佳MSE: {best_mse:.4f}, 当前最佳MAE: {best_mae:.4f}")
        no_improve_epochs = 0  # 重置计数器
    else:
        no_improve_epochs += val_interval
        logging.info(f"当前连续未提升epoch数：{no_improve_epochs}")
        if no_improve_epochs >= 30:
            logging.info(f"指标连续未提升，触发早停机制")
            break

# 保存最终模型和验证结果

torch.save(learn.model.state_dict(), exp_path / "model/patchTST.pth")


# 修改绘图部分：使用自定义指标历史
if len(metrics_history) >= 2:
    epochs = [m[0] for m in metrics_history]
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, [m[1] for m in metrics_history], label='Train Loss')
    plt.plot(epochs, [m[2] for m in metrics_history], label='Valid Loss')
    plt.plot(epochs, [m[3] for m in metrics_history], label='MSE')
    plt.plot(epochs, [m[4] for m in metrics_history], label='MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.savefig(str(exp_path / "metrics/training_metrics.png"))
    plt.close()
else:
    logging.warning("无法生成训练曲线图，数据点不足")
logging.info("训练完成，最终模型和验证结果已保存")