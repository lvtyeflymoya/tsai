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
df_raw = get_long_term_forecasting_data(dsid, target_dir="D:/Python_Project/toolscript", task='S')
# print(df_raw)

# 数据预处理
datetime_col = "date"
freq = 'm'
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
fcst_history = 240 # steps in the past
fcst_horizon = 30  # steps in the future
valid_size   = 0.1  # int or float indicating the size of the validation set
test_size    = 0.2  # int or float indicating the size of the test set

splits = get_long_term_forecasting_splits(df, fcst_history=fcst_history, 
                                          fcst_horizon=fcst_horizon, dsid=dsid, show_plot=False)
# logging.info("分割后的数据内容：")
# logging.info(splits)

# ===== 修改后的检查代码 =====
# 查看划分后的索引范围（修改start/stop的获取方式）
logging.info(f"\n数据划分索引：\n训练集: {len(splits[0])} samples [{splits[0][0]}-{splits[0][-1]}]")
logging.info(f"验证集: {len(splits[1])} samples [{splits[1][0]}-{splits[1][-1]}]")
logging.info(f"测试集: {len(splits[2])} samples [{splits[2][0]}-{splits[2][-1]}]\n")

# 查看具体样本数据（移除未定义的y变量）
sample_idx = 1000  # 查看第几个样本
logging.info("训练集首个样本数据：")
logging.info(f"X_train[0]:\n{df.iloc[splits[0][sample_idx: sample_idx+5]]}")  # 查看前5个样本
# logging.info(f"y_train[0]:\n{y[splits[0][sample_idx: sample_idx+5]]}\n")

# 可视化部分样本（仅显示单个滑动窗口）
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))

# 获取第一个训练样本的窗口
sample_idx = splits[0][1000]
history_slice = slice(sample_idx, sample_idx + fcst_history)
horizon_slice = slice(sample_idx + fcst_history, sample_idx + fcst_history + fcst_horizon)

# 仅绘制目标窗口数据
plt.plot(df.index[history_slice], df[columns[0]].iloc[history_slice],
         color='royalblue', linewidth=2, label=f'History ({fcst_history} steps)')
plt.plot(df.index[horizon_slice], df[columns[0]].iloc[horizon_slice],
         color='darkorange', linestyle='--', linewidth=2, label=f'Horizon ({fcst_horizon} steps)')

# 标注窗口边界
plt.scatter(df.index[sample_idx], df[columns[0]].iloc[sample_idx],
            edgecolor='red', facecolor='none', s=100, linewidth=2, label='Window Start')
plt.scatter(df.index[horizon_slice.start], df[columns[0]].iloc[horizon_slice.start],
            edgecolor='purple', facecolor='none', s=100, linewidth=2, label='Prediction Start')

plt.title(f"Single Sliding Window Visualization (Index {sample_idx})")
plt.legend()
plt.savefig(str(exp_path / "metrics/data_splits_visualization.png"))
plt.close()