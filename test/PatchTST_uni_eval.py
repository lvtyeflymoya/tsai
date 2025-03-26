import sklearn
from tsai.basics import *
from tsai.inference import load_learner
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 创建实验目录
base_dir = Path("trainResult")
existing_exps = [d.name for d in base_dir.glob("experiment*") if d.is_dir()]
exp_numbers = [int(exp[10:]) for exp in existing_exps if exp[10:].isdigit()]
next_exp = max(exp_numbers) + 1 if exp_numbers else 1

exp_path = base_dir / f"experiment{next_exp}"
(exp_path / "metrics").mkdir(parents=True, exist_ok=True)
(exp_path / "model").mkdir(parents=True, exist_ok=True)
(exp_path / "test").mkdir(parents=True, exist_ok=True)


dsid = "ETTh1"
df_raw = get_long_term_forecasting_data(dsid, target_dir="D:/Python_Project/toolscript/csvfile/down_inside", task='S')
# print(df_raw)

datetime_col = "date"
freq = 's'
columns = df_raw.columns[1:]
method = 'ffill'
value = 0

# pipeline
preproc_pipe = sklearn.pipeline.Pipeline([
    ('shrinker', TSShrinkDataFrame()), # shrink dataframe memory usage
    ('drop_duplicates', TSDropDuplicates(datetime_col=datetime_col)), # drop duplicate rows (if any)
    # ('add_mts', TSAddMissingTimestamps(datetime_col=datetime_col, freq=freq)), # ass missing timestamps (if any)
    ('fill_missing', TSFillMissing(columns=columns, method=method, value=value)), # fill missing data (1st ffill. 2nd value=0)
    ], 
    verbose=True)
mkdir('data', exist_ok=True, parents=True)
save_object(preproc_pipe, 'data/preproc_pipe.pkl')
preproc_pipe = load_object('data/preproc_pipe.pkl')

df = preproc_pipe.fit_transform(df_raw)
# print(df)

fcst_history = 1200 # # steps in the past
fcst_horizon = 120  # # steps in the future
valid_size   = 0.1  # int or float indicating the size of the training set
test_size    = 0.2  # int or float indicating the size of the test set

splits = get_long_term_forecasting_splits(df, fcst_history=fcst_history, 
                                          fcst_horizon=fcst_horizon, dsid=None, show_plot=False)
# print(splits)

columns = df.columns[1:]
train_split = splits[0]

# pipeline
exp_pipe = sklearn.pipeline.Pipeline([
    ('scaler', TSStandardScaler(columns=columns)), # standardize data using train_split
    ], 
    verbose=True)
save_object(exp_pipe, 'data/exp_pipe.pkl')
exp_pipe = load_object('data/exp_pipe.pkl')

df_scaled = exp_pipe.fit_transform(df, scaler__idxs=train_split)
# print(df_scaled)


x_vars = df.columns[1:]
y_vars = df.columns[1:]
X, y = prepare_forecasting_data(df, fcst_history=fcst_history, fcst_horizon=fcst_horizon, x_vars=x_vars, y_vars=y_vars)
print(X.shape, y.shape)
# print(X[splits[2]].shape, y[splits[2]].shape)

'''
验证集预测
learn = load_learner('models/patchTST.pt')
scaled_preds, *_ = learn.get_X_preds(X[splits[1]])
scaled_preds = to_np(scaled_preds)
print(f"scaled_preds.shape: {scaled_preds.shape}")

scaled_y_true = y[splits[1]]
results_df = pd.DataFrame(columns=["mse", "mae"])
results_df.loc["valid", "mse"] = mean_squared_error(scaled_y_true.flatten(), scaled_preds.flatten())
results_df.loc["valid", "mae"] = mean_absolute_error(scaled_y_true.flatten(), scaled_preds.flatten())
print(results_df)
'''

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
weights_path = Path("C:/Users/Zhang/Desktop/model/patchTST.pth")
learn = TSForecaster(X, y, splits=splits, batch_size=1024, pipelines=[preproc_pipe, exp_pipe],
                     arch="PatchTST", arch_config=arch_config, metrics=[mse, mae],
                     pretrained=True, weights_path=weights_path)
    
'''测试集预测'''
y_test_preds, *_ = learn.get_X_preds(X[splits[2]])
y_test_preds = to_np(y_test_preds)
print(f"y_test_preds.shape: {y_test_preds.shape}")

# 将numpy数组转换为DataFrame进行反标准化
y_test_preds_df = pd.DataFrame(y_test_preds.reshape(-1, len(columns)), columns=columns)
y_test_preds_inv = exp_pipe.named_steps['scaler'].inverse_transform(y_test_preds_df)
y_test_preds_inv = y_test_preds_inv.to_numpy().reshape(y_test_preds.shape)

y_test = y[splits[2]]
y_test_df = pd.DataFrame(y_test.reshape(-1, len(columns)), columns=columns)
y_test_inv = exp_pipe.named_steps['scaler'].inverse_transform(y_test_df)
y_test_inv = y_test_inv.to_numpy().reshape(y_test.shape)

# 重构连续预测序列（按预测步长间隔取样）
full_preds = y_test_preds_inv[::fcst_horizon].flatten()
full_true = y_test_inv[::fcst_horizon].flatten()

# 生成从0开始递增的时间索引
time_col = pd.RangeIndex(start=0, stop=len(full_true), step=1)

# 保存重构后的预测结果
pd.DataFrame({'true': full_true, 'pred': full_preds}, index=time_col)\
    .rename_axis('date') \
    .to_csv(exp_path / "test/full_predictions.csv")

results_df = pd.DataFrame(columns=["mse", "mae"])
results_df.loc["test", "mse"] = mean_squared_error(y_test.flatten(), y_test_preds.flatten())
results_df.loc["test", "mae"] = mean_absolute_error(y_test.flatten(), y_test_preds.flatten())
print(results_df)
X_test = X[splits[2]]
y_test = y[splits[2]]
plot_forecast(X_test, y_test, y_test_preds, n_samples=5, sel_vars=True)

# 绘制完整预测曲线
plt.figure(figsize=(15, 5))
plt.plot(time_col, full_true, label='True')
plt.plot(time_col, full_preds, label='Predicted')
plt.title('Full Test Set Forecast')
plt.legend()
plt.savefig(exp_path / "test/full_forecast_plot.png")
plt.close()