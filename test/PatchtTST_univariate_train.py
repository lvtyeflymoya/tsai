from tsai.basics import *
import sklearn
import logging

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

# ts = get_forecasting_time_series("Sunspots").values
# X, y = SlidingWindow(60, horizon=1)(ts)
# splits = TimeSplitter(235)(y) 
# tfms = [None, TSForecasting()]
# batch_tfms = TSStandardize()
# fcst = TSForecaster(X, y, splits=splits, path='models', tfms=tfms, batch_tfms=batch_tfms, bs=512, arch="TSTPlus", metrics=mae)
# fcst.fit_one_cycle(5, 1e-3)
# fcst.export("fcst.pkl")

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
    ('add_mts', TSAddMissingTimestamps(datetime_col=datetime_col, freq=freq)), # ass missing timestamps (if any)
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

splits = get_long_term_forecasting_splits(df, fcst_history=fcst_history, fcst_horizon=fcst_horizon, dsid=dsid)
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
learn = TSForecaster(X, y, splits=splits, batch_size=16, path="models", pipelines=[preproc_pipe, exp_pipe],
                     arch="PatchTST", arch_config=arch_config, metrics=[mse, mae])
learn.dls.valid.drop_last = True
logging.info(learn.summary())

# 训练模型
n_epochs = 2
lr_max = 0.0025
learn.fit_one_cycle(n_epochs, lr_max=lr_max)
learn.plot_metrics()
learn.export('../models/patchTST.pt')