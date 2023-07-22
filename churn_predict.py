#%%
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#%%

app_starts = pd.read_csv('churn_prediction/dataset/app starts.txt', sep = "\t")
app_starts_july = pd.read_csv('churn_prediction/dataset/app starts july.txt', sep = "\t")
brochure_views = pd.read_csv('churn_prediction/dataset/brochure views.txt', sep = "\t")
brochure_views_july = pd.read_csv('churn_prediction/dataset/brochure views july.txt', sep = "\t")
installs_df = pd.read_csv('churn_prediction/dataset/installs.txt', sep = "\t")

#%%
df_list = [app_starts, app_starts_july, brochure_views, brochure_views_july, installs_df]
app_starts

#%%
for df in df_list:
    #print(str(df))
    print(df.columns)

#app_starts.info()

#%%
app_starts.columns

#%%

app_starts['dateCreated'] = pd.to_datetime(app_starts['dateCreated'])

#%%
app_starts.info()

#%%

for df in df_list:
    if 'dateCreated' in df.columns:
        df['dateCreated'] = pd.to_datetime(df['dateCreated'])
        
        
#%%

brochure_views.info()

#%%
#installs_df.info()  

installs_df['InstallDate'] = pd.to_datetime(installs_df['InstallDate'])

#%%

app_starts

#%%

brochure_views

#%%
brochure_views.shape     

#%%
brochure_views.columns 

#%%
brochure_views['dateCreated'].nunique()

#%%
brochure_views_july.shape

#%%
brochure_views_july['id'].nunique()

#%%

installs_df.columns

#%%
installs_df.shape

#%%
installs_df['productId'].nunique()

#%%

installs_df.head()

#%%


px.line(data_frame=brochure_views, x='dateCreated', y='page_turn_count')

#%%

brochure_views['dateCreated'].dt.day

#%%

brochure_all = brochure_views.merge(brochure_views_july, how='outer')

#%%

brochure_all['dateCreated'].max()

#%%
brochure_all['dateCreated'].min()

#%%

app_starts_all = app_starts.merge(app_starts_july, how='outer')

#%%
app_starts_all['dateCreated'].min()

#%%

brochure_all.head()

#%%

app_starts_all['dateCreated'].max()


#%%

brochure_all['hours'] = brochure_all['dateCreated'].dt.hour

#%%

brochure_all['day'] = brochure_all['dateCreated'].dt.day

#%%
brochure_all.shape

#%%

brochure_all.groupby(by=['dateCreated', 'hours'])['page_turn_count'].mean().reset_index()

#%%

brochure_all.groupby(by=['dateCreated','hours', 'day'])['page_turn_count'].sum().reset_index()


#%%

brochure_all.groupby([brochure_all['dateCreated'].dt.hour])['page_turn_count'].sum()

#%%

brochure_all_hourly = brochure_all.resample('60min', on='dateCreated').sum()

#%%

installs_df.columns

#%%

px.line(data_frame=brochure_all_hourly, y='view_duration')


#%% prep data for prophet 

#brochure_reset_date = 

brochure_prep = (brochure_all_hourly.reset_index()
                    .rename(columns={'dateCreated': 'ds', 'page_turn_count': 'y'})
                    [['ds','y']]
                    
                )



#%%

from prophet import Prophet

from prophet.plot import plot_seasonality

#%%
model = Prophet(seasonality_mode='multiplicative')
model.fit(brochure_prep)

#%%
future = model.make_future_dataframe(periods=3*24, freq='h')

forecast = model.predict(future)
fig = model.plot(forecast)
plt.show()
fig2 = model.plot_components(forecast)
plt.show()


#%% ########   cross validation of model ###########
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation

#%%
add_changepoints_to_plot(fig.gca(), model, forecast)
plt.show()

#%% crpss val

df_cv = cross_validation(model, 
                         horizon='30 days'#, parallel='processes'
                         )

#%%

df_cv.tail()


#%% ####### evaluating performance metrics #####
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric


#%%

df_p = performance_metrics(df_cv)

#%%
df_p


#%%

plot_cross_validation_metric(df_cv, metric='rmse')

#%% #######  hyperparameter optimization grid seach ########

param_grid = {'changepoint_prior_scale': [0.5, 0.1, 0.01, 0.001],
              'seasonality_prior_scale': [10.0, 1, 0.1, 0.01],
              'seasonality_mode': ['additive', 'multiplicative']
              }

import numpy as np
import itertools

all_params = [dict(zip(param_grid.keys(), value))
              for value in itertools.product(*param_grid.values())]
rmse_values = []

#%%
for params in all_params:
    model = Prophet(**params).fit(brochure_prep)
    df_cv = cross_validation(model, horizon='30 days')
    df_p = performance_metrics(df_cv, rolling_window=1)
    rmse_values.append(df_p['rmse'].values[0])
     

#%%

results = pd.DataFrame(all_params)
results['rmse'] = rmse_values
results.head()

#%%

best_params = all_params[np.argmin(rmse_values)]
print(best_params)

#%%


#%%
add_changepoints_to_plot(fig.gca(), model, forecast)
plt.show()


#%% #############  include productId and view_duration  ##############
brochure_install_all = brochure_all.merge(installs_df, how='outer', on='userId')

#%%

len(installs_df['userId'].isin(brochure_all['userId']))

#%%
installs_df['userId'].nunique()

#%%

brochure_install_all.info()

#%%

brochure_all[brochure_all['page_turn_count']==0]

#%%

brochure_install_all_subset = (brochure_install_all[['userId', 'dateCreated', 
                                              'view_duration', 'productId',
                                              'page_turn_count'
                                              ]]
                                .dropna(subset=['dateCreated'])
                                )

#%%

brochure_install_dummy = pd.get_dummies(brochure_install_all_subset, 
                                        columns=['productId'], 
                                       # drop_first=True
                                        )

#%%
brochure_install_dummy.resample('60min', on='dateCreated').sum()#['productId_de-kaufda-android'].sum()

#%%

brochure_install_hourly = brochure_install_all_subset.resample('60min', on='dateCreated').sum().reset_index()#.agg({'productId': 'count', 
                                                                    #  'page_turn_count': 'sum',
                                                                    #  'view_duration': 'sum'
                                                                    #  }
                                                                    # )


#%%

brochure_install_hourly

#%%
px.line(data_frame=brochure_install_hourly, 
        x='dateCreated', 
        y='page_turn_count',
        template='plotly_dark'
        )

#%%
px.line(data_frame=brochure_install_hourly, 
        x='dateCreated', 
        y='view_duration',
        template='plotly_dark'
        )


#%%

brochure_install_hourly.plot(x='dateCreated', y='view_duration')
brochure_install_hourly.plot(x='dateCreated', y='page_turn_count')

#%% #################   including view_duration   #####################

brochure_install_prep = (brochure_install_hourly.reset_index()
                            .rename(columns={'dateCreated': 'ds', 'page_turn_count': 'y'})
                            [['ds','y','view_duration']]
                            
                        )


brochure_install_prep

#%% select up june for training set, leave july for testing

train_df = brochure_install_prep[brochure_install_prep['ds'].dt.month < 7]#.tail(5)

#%%

model_m = Prophet(seasonality_mode='multiplicative')
model_m.add_regressor('view_duration')
model_m.fit(train_df)

#%%
future_m = model_m.make_future_dataframe(periods=31*24, freq='h')

#%%
future_m['view_duration'] = brochure_install_prep['view_duration']

#%%

#future_m.tail()
#%%
forecast_m = model_m.predict(future_m)

#%%
fig_m = model_m.plot(forecast_m)
plt.show()
fig2_m = model_m.plot_components(forecast_m)
plt.show()


#%%  ### cross validation #####
model_cvm = Prophet(seasonality_mode='multiplicative')
model_cvm.add_regressor('view_duration')
model_cvm.fit(brochure_install_prep)

#%%
#add_changepoints_to_plot(fig_m.gca(), model_cvm, forecast_m)
#plt.show()

#%% crpss val

df_cvm = cross_validation(model_cvm, 
                            horizon='30 days'#, parallel='processes'
                         )

#%%

df_cvm#.tail()

#%%

df_cvm_perfmetric = performance_metrics(df_cvm)

#%%
df_cvm_perfmetric


#%%

plot_cross_validation_metric(df_cvm, metric='rmse')

#%% #######  hyperparameter optimization grid seach ########

param_grid_m = {'changepoint_prior_scale': [0.5, 0.1, 0.01, 0.001],
              'seasonality_prior_scale': [10.0, 1, 0.1, 0.01],
              'seasonality_mode': ['additive', 'multiplicative']
              }

import numpy as np
import itertools

all_params_m = [dict(zip(param_grid_m.keys(), value))
              for value in itertools.product(*param_grid_m.values())]
rmse_values_m = []

#%%
for params in all_params_m:
    model = Prophet(**params)
    model.add_regressor(name='view_duration')
    model.fit(brochure_install_prep)
    df_cv = cross_validation(model, horizon='30 days')
    df_p = performance_metrics(df_cv, rolling_window=1)
    rmse_values_m.append(df_p['rmse'].values[0])
     

#%%

results_m = pd.DataFrame(all_params_m)
results_m['rmse'] = rmse_values_m

#%%
results_m#.head()

#%%

best_params_m = all_params_m[np.argmin(rmse_values_m)]
print(best_params_m)


#%% build model with best parameters

best_params_model_m = Prophet(**best_params_m)
best_params_model_m.add_regressor(name='view_duration')
best_params_model_m.fit(brochure_install_prep)

#%%

best_params_cv_res = cross_validation(best_params_model_m, 
                                            horizon='30 days'#, parallel='processes'
                                        )
#%%
best_param_perfmetric = performance_metrics(best_params_cv_res)

#%%
best_param_perfmetric


#%%

plot_cross_validation_metric(best_params_cv_res, metric='rmse')

#%% saving the model

from prophet.serialize import model_to_json, model_from_json
import json

#%% save model as json

with open('page_turn_count_model.json', 'w') as file_out:
    json.dump(model_to_json(best_params_model_m),file_out )


#%% loading the model

with open('page_turn_count_model.json', 'r') as file_in:
    model_loaded = model_from_json(json.load(file_in))

#%%

loaded_model_forecast = model_loaded.predict()

model_loaded.plot(loaded_model_forecast)

#%%
brochure_install_dummy.columns

#%%

brochure_install_prep

#%%

train_df = brochure_install_prep[brochure_install_prep['ds'].dt.month < 7]

#%%
test_df = brochure_install_prep[brochure_install_prep['ds'].dt.month > 6]


#%%
train_df.shape

#%%
brochure_install_prep['ds'].dt.month

#%%

train_df = brochure_install_prep[brochure_install_prep['ds'].dt.month < 7].copy()

#%%

test_df = brochure_install_prep[brochure_install_prep['ds'].dt.month > 6]
test_df.shape

#%%
len(train_df) / len(brochure_install_prep)


#%% Deep learning for time series forecasting

from time_series.dataset.time_series import TrainingDataSet

#%%
from time_series.utils import evaluate_model
from time_series.models.deepar import DeepAR
from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow as tf


disable_eager_execution()  # for graph mode
tf.compat.v1.experimental.output_all_intermediates(True)
#%%
#train_df = get_energy_demand()

brochure_install_deepl_prep = brochure_install_prep.set_index(keys='ds')

#%%
tds = TrainingDataSet(brochure_install_deepl_prep, train_split=0.75)

#%%
N_EPOCHS = 100

#%%



#%%

ar_model = DeepAR(tds)
ar_model.instantiate_and_fit(verbose=1, epochs=N_EPOCHS)

#%%
y_predicted = ar_model.model.predict(tds.X_test)
evaluate_model(tds=tds, y_predicted=y_predicted, columns=brochure_install_deepl_prep.columns, first_n=10)

#%%
ar_model.model

#%%  #####  N-BEATS    ############

from time_series.models.nbeats import NBeatsNet

nb = NBeatsNet(tds)
nb.instantiate_and_fit(verbose=1, epochs=N_EPOCHS)
#y_predicted = nb.model.predict(tds.X_test, steps=10)
#evaluate_model(first_n=10)
print(nb.model.evaluate(tds.X_test, tds.y_test))

#%%

y_predicted = nb.model.predict(tds.X_test)
evaluate_model(tds=tds, y_predicted=y_predicted, columns=brochure_install_deepl_prep.columns, first_n=10)

#%%
from sklearn.metrics import mean_squared_error

#%%   ######### LSTM not working #########
# from time_series.models.LSTM import LSTM
# lstm = LSTM(tds)
# #lstm.instantiate_and_fit(verbose=1, epochs=100)

# lstm.fit(verbose=1, epochs=100)
# #y_predicted = lstm.model.predict(tds.X_test, steps=10)
# #evaluate_model(first_n=10)
# print(lstm.model.evaluate(tds.X_test, tds.y_test))


#%% ######### TRANSFORMER  ######
# only training a 1-step prediction here:
from time_series.models.transformer import Transformer

trans = Transformer(tds)
trans.instantiate_and_fit(verbose=1, epochs=N_EPOCHS)
print(trans.model.evaluate(tds.X_test, tds.y_test))

#%%
y_predicted = trans.model.predict(tds.X_test)  # .reshape(-1, 10)
evaluate_model(tds=tds, y_predicted=y_predicted, columns=brochure_install_deepl_prep.columns, first_n=10)


#%% tcn

from time_series.models.TCN import TCNModel
tcn_model = TCNModel(tds)
tcn_model.instantiate_and_fit(verbose=1, epochs=100)
print(tcn_model.model.evaluate(tds.X_test, tds.y_test))


#%%

y_predicted = tcn_model.model.predict(tds.X_test)
evaluate_model(tds=tds, y_predicted=y_predicted,
               columns=brochure_install_deepl_prep.columns, 
               first_n=10
               )



#%% ############ GLUONTS ######

from gluonts.dataset.pandas import PandasDataset


#%%
train_ds = PandasDataset.from_long_dataframe(train_df, target='y', freq='H', item_id='ds')

#%%
from gluonts.torch.model.deepar import DeepAREstimator
#%%
estimator = DeepAREstimator(freq='H', prediction_length=24*30, 
                            num_layers=3, 
                            trainer_kwargs={'max_epochs':30}
                            )

predictor = estimator.train(train_ds)



##########################
#%%  deep learning from scratch

import tensorflow.keras as keras
import tensorflow as tf

DROPOUT_RATIO = 0.2
HIDDEN_NEURONS = 10

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

def create_model(passengers):
  input_layer = keras.layers.Input(len(passengers.columns))

  hiden_layer = keras.layers.Dropout(DROPOUT_RATIO)(input_layer)
  hiden_layer = keras.layers.Dense(HIDDEN_NEURONS, activation='relu')(hiden_layer)

  output_layer = keras.layers.Dropout(DROPOUT_RATIO)(hiden_layer)
  output_layer = keras.layers.Dense(1)(output_layer)

  model = keras.models.Model(inputs=input_layer, outputs=output_layer)

  model.compile(loss='mse', optimizer=keras.optimizers.Adagrad(),
    metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError()])
  return model


#%%
brochure_only_y = brochure_install_deepl_prep[['y']]

#%%
model = create_model(brochure_only_y)

#%%

model.fit()




#%%
train_df.rename_index({})



#%%
tds.train_split

#%%
_

#%%

brochure_install_hourly.set_index(keys='dateCreated', inplace=True)




#%%  ##########  LSTM univariante single step style

import tensorflow as tf
from sklearn import preprocessing

#%%

tf.random.set_seed(2023)
np.random.seed(2023)

#%%

validate = brochure_install_prep['y'].tail(48)

df_without_validate = brochure_install_prep.drop(validate.index)

#%%
uni_data = df_without_validate['y']
uni_data.index = df_without_validate['ds']
uni_data.head()

#%%

df_without_validate.info()

#%%
uni_data = uni_data.values
scaler_x = preprocessing.MinMaxScaler()

x_rescaled = scaler_x.fit_transform(uni_data.reshape(-1, 1))

#%%
def custom_ts_univariate_data_prep(dataset, start, end, window, horizon):
    X = []
    y = []
    
    start = start + window
    if end is None:
        end = len(dataset) - horizon
        
    for i in range(start, end):
        indicesx = range(i - window, i)
        X.append(np.reshape(dataset[indicesx], (window, 1)))
        indicesy = range(i, i + horizon)
        y.append(dataset[indicesy])
    return np.array(X), np.array(y)


#%%

univar_hist_window = 48
horizon = 1
TRAIN_SPLIT = 2100

x_train_uni, y_train_uni = custom_ts_univariate_data_prep(x_rescaled, 0, 
                                                          TRAIN_SPLIT,
                                                          univar_hist_window, 
                                                          horizon
                                                          )
x_val_uni, y_val_uni = custom_ts_univariate_data_prep(x_rescaled, 
                                                      TRAIN_SPLIT, None,
                                                      univar_hist_window, horizon
                                                      )

#%%
x_train_uni

y_train_uni



#%%

BATCH_SIZE = 256
BUFFER_SIZE = 150

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()


#%%
model_path = "LSTM_Univarient_brochure.h5"

#%%

lstm_model = tf.keras.models.Sequential(
                                        [tf.keras.layers.LSTM(100, input_shape=x_train_uni.shape[-2:], return_sequences=True),
                                        tf.keras.layers.Dropout(0.2),
                                        tf.keras.layers.LSTM(units=50, return_sequences=False),
                                        tf.keras.layers.Dropout(0.2),
                                        tf.keras.layers.Dense(units=1)
                                        ]
                                    )

lstm_model.compile(optimizer='adam', loss='mse')

#%%

EVALUATION_INTERNAL = 100
EPOCHS = 150
history = lstm_model.fit(train_univariate, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERNAL,
                         validation_data=val_univariate,
                         validation_steps=50, verbose=1,
                         callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                                     patience=10, verbose=1, mode='min'
                                                                     ),
                                    tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss',
                                                                       save_best_only=True, mode='min',
                                                                       verbose=0)
                                    ]
                         )

#%%

trained_model = tf.keras.models.load_model(model_path)
trained_model.summary()

#%%

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'], loc='upper left')
plt.rcParams['figure.figsize'] = [16, 9]
plt.show()

#%%
uni = brochure_install_prep['y']
validatehori = uni.tail(48)
validatehist = validatehori.values
result = []

# define forecast len
window_len = 48
val_rescaled = scaler_x.fit_transform(validatehist.reshape(-1, 1))

for i in range(1, window_len+1):
    val_rescaled = val_rescaled.reshape((1, val_rescaled.shape[0], 1))
    predicted_results = trained_model.predict(val_rescaled)
    print(f'predicted: {predicted_results}')
    result.append(predicted_results[0])
    val_rescaled = np.append(val_rescaled[:,1:], [[predicted_results]])
    print(val_rescaled)
    
    
#%%
result_inv_trans = scaler_x.inverse_transform(result)
result_inv_trans


#%%
from sklearn import metrics

def timeseries_evaluation_metrics_func(y_true, y_pred):
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print('Evaluation metric results:-')
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true, y_pred)}', end='\n\n')
 
    
#%%
timeseries_evaluation_metrics_func(validate, result_inv_trans)

#%%
plt.plot(list(validate))
plt.plot(list(result_inv_trans))
plt.title("Actual vs Predicted")
plt.ylabel("Traffic volume")
plt.legend(("Actual", "predicted"))
plt.show()
    




#%% ##################### univariate horizon style  ##########################

long_horizon = 10

lstm_model_lg_hori = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(100, input_shape=x_train_uni.shape[-2:], return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(units=50, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=long_horizon)
])

lstm_model_lg_hori.compile(optimizer='adam', loss='mse')

model_path = "horizon_style_LSTM_Univarient.h5"

#%%

history_lg_hori = lstm_model_lg_hori.fit(train_univariate, epochs=EPOCHS, 
                                         steps_per_epoch=EVALUATION_INTERNAL,
                                            validation_data=val_univariate,
                                            validation_steps=50, verbose=1,
                                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                                                        patience=10, verbose=1, mode='min'
                                                                                        ),
                                                        tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss',
                                                                                        save_best_only=True, mode='min',
                                                                                        verbose=0)
                                                        ]
                                            )


#%% load model

trained_model_lg_hori = tf.keras.models.load_model(model_path)
# show model architecture
trained_model_lg_hori.summary()

#%% plot model loss history

def plot_loss_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'validation loss'], loc='upper left')
    plt.rcParams['figure.figsize'] = [16, 9]
    return plt.show()
        
plot_loss_history(history=history_lg_hori)

#%% make prediction

uni_lg = brochure_install_prep['y']
validatehori_lg = uni_lg.tail(10)
validatehist_lg = validatehori_lg.values

scale_x_lg = preprocessing.MinMaxScaler()
val_rescaled_lg = scale_x_lg.fit_transform(validatehist_lg.reshape(-1, 1))
val_rescale_lg = val_rescaled_lg.reshape((1, val_rescaled_lg.shape[0], 1))

predicted_results_lg = trained_model_lg_hori.predict(val_rescaled_lg)
predicted_inver_res_lg = scale_x_lg.inverse_transform(predicted_results_lg)
predicted_inver_res_lg

#%%
timeseries_evaluation_metrics_func(validatehist_lg, predicted_inver_res_lg[0])

#%%
plt.plot(list(validatehist_lg))
plt.plot(list(predicted_inver_res_lg[0]))
plt.title('Actual vs Predicted')
plt.ylabel('Page turn count')
plt.legend(("Actual", "Predicted"))
plt.show()





#%%  ######## Bidirectional LSTM Univariate  ###################


bi_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=True),
                                  input_shape=x_train_uni.shape[-2:]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50)),
    tf.keras.layers.Dense(20, activation='softmax'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=1)
])

bi_lstm_model.compile(optimizer='adam', loss='mse')

bi_model_path_sg_hori = "bidirectional_LSTM_sg_hori.h5"


#%%
bilstm_history_sghori = bi_lstm_model.fit(train_univariate, epochs=20, 
                                         steps_per_epoch=EVALUATION_INTERNAL,
                                            validation_data=val_univariate,
                                            validation_steps=50, verbose=1,
                                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                                                        patience=10, verbose=1, mode='min'
                                                                                        ),
                                                        tf.keras.callbacks.ModelCheckpoint(bi_model_path_sg_hori, monitor='val_loss',
                                                                                        save_best_only=True, mode='min',
                                                                                        verbose=0)
                                                        ]
                                            )
#%%
trained_model_bilstm_sghori = tf.keras.models.load_model(bi_model_path_sg_hori)
trained_model_bilstm_sghori.summary()

#%%
plot_loss_history(bilstm_history_sghori)

#%%
# validate data is last 10 hours of all data == test date
validate = brochure_install_prep['y'].tail(10)

# remaining data after validation data taken
uni = brochure_install_prep.drop(validate.index)  #['y']#

# take target and reset index to date column
uni_data = uni['y']
uni_data.index = uni['ds']

# validatehori is last 48 hours of training data
validatehori = uni_data.tail(48)
validatehist = validatehori.values

# scale validatehori
scaler = preprocessing.MinMaxScaler()

val_rescaled = scaler.fit_transform(validatehist.reshape(-1, 1))

#%%
def predict_single_step_horizon(model, preprocessed_x, 
                                forecast_window_len=10):
    result = []
    for i in range(1, forecast_window_len+1):
        val_rescaled = preprocessed_x.reshape((1, preprocessed_x.shape[0], 1))
        predicted_results = model.predict(val_rescaled)
        result.append(predicted_results[0])
        
    return result


results = predict_single_step_horizon(model=trained_model_bilstm_sghori, 
                            preprocessed_x=val_rescaled,
                            forecast_window_len=10
                            )

result_inv_trans = scaler.inverse_transform(results)

#%%
timeseries_evaluation_metrics_func(validate, result_inv_trans)

#%%

plt.plot(list(validate))
plt.plot(list(result_inv_trans))
plt.title("Actual vs Predicted")
plt.ylabel("page turn count")
plt.legend(('Actual', 'Predicted'))
plt.show()





#%%  ############  multivariate time series ######################
##### LSTM Horizon style  #######

# create weekday feature, hour, day of week, day of month, weekend feature
brochure_install_prep

brochure_install_prep_transform = brochure_install_prep.copy()


#%% hour

brochure_install_prep_transform['day_of_week'] = brochure_install_prep_transform['ds'].dt.day_of_week
brochure_install_prep_transform['hour'] = brochure_install_prep_transform['ds'].dt.hour
brochure_install_prep_transform['month_day'] = brochure_install_prep_transform['ds'].dt.day
brochure_install_prep_transform['weekend'] = (np.where(brochure_install_prep_transform['ds']
                                             .dt.day_name().isin(["Saturday", "Sunday"]), 
                                             1, 0
                                             )
                                    )


#%% # take last 10 hour as validation set

validate_multivar = brochure_install_prep_transform.tail(10)
#brochure_install_prep_transform[2:].tail(10)

#%%
train_multivar = brochure_install_prep_transform.drop(validate_multivar.index)


#%%

multi_scaler = preprocessing.StandardScaler()

#%%
columns_to_scale = brochure_install_prep_transform.columns[2:-1]

#%%

multi_scaler.fit(brochure_install_prep_transform[columns_to_scale])

#%%
brochure_install_prep_transform[columns_to_scale] = multi_scaler.transform(brochure_install_prep_transform[columns_to_scale])

#%%

y_scaler = preprocessing.StandardScaler()

brochure_install_prep_transform['y'] = y_scaler.fit_transform(brochure_install_prep_transform[['y']])

#%%
brochure_install_prep_transform

#%% # take last 10 hour as validation set

validate_multivar = brochure_install_prep_transform.tail(20)
#brochure_install_prep_transform[2:].tail(10)

#%%
train_multivar = brochure_install_prep_transform.drop(validate_multivar.index)


# .copy().set_index(keys='ds')





#%%


preprocessing.MinMaxScaler().fit_transform(brochure_install_prep_transform[['y']])

#%%
brochure_install_prep[['view_duration_scaled', 
                       'day_of_week_scaled', 'hour_scaled', 'month_day_scaled']] = multi_scaler.fit_transform(brochure_install_prep[['view_duration', 'day_of_week', 'hour', 'month_day']])


#%%

brochure_install_prep.drop(columns=['view_duration_scaled', 'day_of_week_scaled', 
                                    'hour_scaled', 'month_day_scaled'],
                           inplace=True
                           )

#%%

brochure_install_prep_transform = brochure_install_prep.copy()

#%%

brochure_install_prep.columns[2:-5].values


#%%

#brochure_install_prep['dayname'] = brochure_install_prep['ds'].dt.day_name()


#brochure_install_prep.drop(columns='dayname', inplace=True)
#%%

#[brochure_install_prep[validate.index]]








#%%

# (brochure_install_dummy.resample('60min', on='dateCreated')
#  .agg({#'productId': 'count', 
#        'page_turn_count': 'sum',
#        'view_duration': 'sum',
#        'productId_com-bonial-kaufda': 'count',
#        'productId_de-kaufda-android': 'count',
#        'productId_de.kaufda.kaufda': 'count'
       
       
#        }
#       )


# )




#%%
#brochure_all_hourly = brochure_all.resample('60min', on='dateCreated').sum()



#%%
#plot_seasonality(model, 'daily', figsize=(10, 3))
#plt.show()

#%%
## Exploratory analysis to do for time series
# viz page_turn_count and view_duration and show positive relation



#%%
"""
## Options

1. aggregate the data to hourly

"""

#%%

"""
Installs are churn

########   brochure_views_july  ##########

target var  = 'page_turn_count',
pred var = 'view_duration',

#######  installs_df   ########
-- productId


join brochure_views_july to brochure_view


join brochure_views to install based on userId
"""





# %%
