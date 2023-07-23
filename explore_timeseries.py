#%%
"""
Premise

Prior to building models, exploratory data analysis is undertaken to 
understand the data better using descriptive statistics among others.
To do this the various the datasets are read and attempt is made to understand
the variables. First all modules needed for the analysis are imported 
followed by reading the datasets.txt

"""
#%%
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_seasonality
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
import numpy as np
import itertools
from prophet.serialize import model_to_json, model_from_json
import json



#%%
"""
All the dataset are read as follows:
"""

#%%

app_starts = pd.read_csv('churn_prediction/dataset/app starts.txt', sep = "\t")
app_starts_july = pd.read_csv('churn_prediction/dataset/app starts july.txt', sep = "\t")
brochure_views = pd.read_csv('churn_prediction/dataset/brochure views.txt', sep = "\t")
brochure_views_july = pd.read_csv('churn_prediction/dataset/brochure views july.txt', sep = "\t")
installs_df = pd.read_csv('churn_prediction/dataset/installs.txt', sep = "\t")

#%% Exploratory data analysis
"""
The first exploration done on the dataset is to identify the various features, their 
data types and presence of missing values. The data types of features determines the 
type visualization and preprocessing techniques to apply to prepare the data for 
machine learning. For example, onehot encoding for categorical features and scaling 
for numerical features among others.

"""

#%%
## to reduce repetition of same analysis for all dataset a loop is used
df_list = [app_starts, app_starts_july, brochure_views, brochure_views_july, installs_df]

for df in df_list:
    df.info()

#%%
"""
Among others, id and userId are features in the dataset that are more of identifiers for the 
data and will prove useful for joining the dataset. These features however are not 
likely to provide predictive signals for modeling hence not explore for that 
purpose. userId is a feature in all the dataset hence used for joining them.

Another observation is that the dataset has dateCreated and InstallDate features 
which suggests the dataset is likely to exhibit the properties of time series. 

The data type of dateCreated and InstallDate is however not datetime as will 
be expected hence data type conversion will be undertaken.
"""

#%% convert dateCreated to datetime data type

for df in df_list:
    if 'dateCreated' in df.columns:
        df['dateCreated'] = pd.to_datetime(df['dateCreated'])

#%%

installs_df['InstallDate'] = pd.to_datetime(installs_df['InstallDate'])
 
 #%%
 
"""
From enquiries made, page_turn_count variable is interpreted as churn and hence identifed
as the target variable for the task.

The brochure views dataset contains the page_turn_count variable hence identifed as important 
for the analysis. It is possible that the brochure views july is meant to 
be a test dataset but will be merged with the brochure views dataset 
for exploratory analysis. install dataset also has features that needs to 
be explored to model hence will also be merged with the other other datasets.

Thus, the dataset identified as likely to be relevant for the model building 
are brochure views, brochure views july, and installs.

app starts and app starts july only contain identifiers which will not 
be relevant for the modeling task hence not used.
"""
 
#%% merging datasets

brochure_all = brochure_views.merge(brochure_views_july, how='outer')

brochure_install_all = brochure_all.merge(installs_df, how='outer', on='userId')

#%% Categorical data exploration

"""
A key consideration for analyzing categorical features is cardinality. Generally,
categorical features with high cardinality when preprocessed with techniques 
such as one hot encoding leads to expotential growth in dimensionality. Despite 
there are techniques such as recategorizing for treating high cardinal features, 
this task will employ eliminating very high cardinal features for modeling.
Among other reasons for this, is the fact very little known about these 
features to devise an appropriate strtegy for recategorization.

The function for checking the cardinality of variables is defined as follows:  

"""

#%%

categorical_vars = ['brochure_id', 'productId', 'model', 'campaignId']

def get_number_of_unique_values(data: pd.DataFrame, variable: str):
    num_values = data[variable].nunique()
    print(f'{variable} has {num_values} unique values')
    
    

for cat_var in categorical_vars:
    get_number_of_unique_values(data=brochure_install_all, variable=cat_var)   


#%%

# from the cardinality check, only productId is of a low cardinal with 
# 3 unique values. Hence, productId is the only categorical variable to 
# explored further.

#%%

""" Data visualization
Visualization of page_turn_count as the target variable is undertaken as
follows:
"""

#%%

plotly_fig = px.line(data_frame=brochure_install_all, 
                    x='dateCreated', 
                    y='page_turn_count'
                    )

#%%
plotly_fig.show()

#%%
"""
The time series plot of page_turn_count indicates the time scale for 
the data to be of a very high temporal resolution in addition
to the fact that interval for the data collection is irregular.

Thus, the data needs to be preprocessed to a time scale of equal time interval 
to ensure consistent prediction. To achieve this, the data is resampled 
to hourly intervals. By this, the total page_turn_count for every hour 
is estimated as the target variable to be predicted.

This approach presents a challenge for handling productId feature. 
productId is not numeric but nominal hence it is not possible to 
aggregate it at hourly interval as done for page_turn_count. Moreover,
counting the total number of productId as a feature will not add any insight 
given that it will essentially produce the number of data points recorded 
in an hour interval. Thus, only numeric features can be further explored
for modeling, and for this, view_duration will be used.

First, the dataset is resampled as follows:


"""

#%%

brochure_install_all_subset = (brochure_install_all[['dateCreated', 'view_duration', 
                                                    'page_turn_count'
                                              ]]
                                .dropna(subset='dateCreated')
                                )
 
brochure_install_hourly = (brochure_install_all_subset
                           .resample('60min', on='dateCreated')
                           .sum().reset_index()
                           )

#%% 
"""The resulting hourly aggregation of page_turn_count and view_duration 
are visualized as follows:
"""

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
plt.figure(figsize=(15,15))
plt.plot(brochure_install_hourly['dateCreated'], 
        brochure_install_hourly['page_turn_count'],
        
        )
plt.title('Page turn count per hour')
plt.ylabel('Total paget turn count')
plt.xlabel('date time (hourly)')


#%%
def lineplot(data, y_var, x_var, title, ylabel, xlabel,
             width=15, height=15):
    plt.figure(figsize=(width,height))
    plt.plot(data[x_var], 
            data[y_var],
            
            )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    return plt.show()

#%%

lineplot(data=brochure_install_hourly, 
            x_var='dateCreated', 
            y_var='page_turn_count',
            ylabel='Total paget turn count',
            xlabel='date time (hourly)',
            title='Page turn count per hour'
        )
#%%
lineplot(data=brochure_install_hourly, 
            x_var='dateCreated', 
            y_var='view_duration',
            ylabel='Total view_duration',
            xlabel='date time (hourly)',
            title='Total view duration per hour'
        )

#%%
"""
From the two plots, periods of high and low page turn count
correspond with those of high and low view duration respectively
hence view duration is likely to provide predictive signals 
for forecasting page turn count. Thus, view duration will be 
included as a predictor for the forecasting.

In order to understand the dataset better in terms of its 
time series properties, it is decompose into various components as 
trend, seasonality and residuals as follows:
"""

#%%

brochure_install_hourly.set_index(keys='dateCreated', inplace=True)

#%%
brochure_install_hourly
    
# %%
from statsmodels.tsa.seasonal import seasonal_decompose
# decomposition = seasonal_decompose(brochure_install_hourly['page_turn_count'])

# trend = decomposition.trend
# seasonal = decomposition.seasonal
# residual = decomposition.resid

# fig = plt.figure(figsize=(10,10))
# ax1 = fig.add_subplot(411)
# ax2 = fig.add_subplot(412)
# ax3 = fig.add_subplot(413)
# ax4 = fig.add_subplot(414)
# ax1.set_title('Original Time Series - page_turn_count')
# ax1.plot(brochure_install_hourly['page_turn_count'])
# ax2.set_title('Trend')
# ax2.plot(trend)
# ax3.set_title('Seasonality')
# ax3.plot(seasonal)
# ax4.set_title('Residuals')
# ax4.plot(residual)
# plt.tight_layout()
# plt.show()


# %%

def decompose_timeseries(data, variable_to_decompose,
                         plot_width = 15, plot_height = 15):
    decomposition = seasonal_decompose(data[variable_to_decompose])

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    fig = plt.figure(figsize=(plot_width,plot_height))
    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)
    ax1.set_title(f'Original Time Series - {variable_to_decompose}')
    ax1.plot(data[variable_to_decompose])
    ax2.set_title('Trend')
    ax2.plot(trend)
    ax3.set_title('Seasonality')
    ax3.plot(seasonal)
    ax4.set_title('Residuals')
    ax4.plot(residual)
    plt.tight_layout()
    return plt.show()


#%%

decompose_timeseries(data=brochure_install_hourly, variable_to_decompose='page_turn_count')

#%%
decompose_timeseries(data=brochure_install_hourly, 
                     variable_to_decompose='view_duration',
                     
                     )

#%%

"""from the decomposition, it is obvious view duration is experiencing the same 
trend and seasonality as page_turn_count hence can be used for prediction.
Moreover, the trend is multiplicative. This is an important insight gained for 
modeling purpose.

From the exploratory analysis, the the type of trend (multiplicative), 
relevant predictors (view duration) were determined. Also, the data was resampled
to hourly scale to ensure regular and consistent time interval for forecasting.

This provide the needed impetus for modeling. First, prophet model will be 
used. This is basically a linear regression model for time series forecasting.
Prophet uses probalilistic approach to modeling.


"""

#%%  ###  Forecasting with prophet model  ####
## Data preparation
"""
The data is preprocessed for use in prophet model as follows:
"""

#%%

brochure_install_prep = (brochure_install_hourly.reset_index()
                            .rename(columns={'dateCreated': 'ds', 'page_turn_count': 'y'})
                            [['ds','y','view_duration']]
                            
                        )


#%%

"""The model is then defined first with default parameters except changing 
the seasonality mode to multiplicative as identified for exploratory analysis.

Cross validation is used to evaluate the model.

"""

#%%

#%%  ### cross validation #####
# initialize model
model_cvm = Prophet(seasonality_mode='multiplicative')
# add predictor to the model
model_cvm.add_regressor('view_duration')
# fit the model on the preprocessed data
model_cvm.fit(brochure_install_prep)

#%%

"""
for cross validation, a 30 days horizon is used. This is 
an hourly forecast of page turn count for 24 hours for 30 days.
"""

#%%
df_cvm = cross_validation(model_cvm, 
                            horizon='30 days'#, parallel='processes'
                         )


#%%
df_cvm

#%%

"""
Several evaluation emetrics were computed. The result for model with 
default parameters is a Root Mean Squared Error (RMSE) of 367.030746
"""

#%%
df_cvm_perfmetric = performance_metrics(df_cvm, rolling_window=1)
df_cvm_perfmetric


#%%

plot_cross_validation_metric(df_cvm, metric='rmse')

#%%
"""
The graph above is visual representation of the cross validation
results of the model.

Next, the model is optimized through hyperparameter optimization 
using the grid search approach. 

Due to the computational expensiveness of undertaking grid search,
only a few parameters are specified. This is implemented as follows:

"""

#%% #######  hyperparameter optimization grid seach ########

param_grid_m = {'changepoint_prior_scale': [0.5, 0.1, 0.01, 0.001],
              'seasonality_prior_scale': [10.0, 1, 0.1, 0.01],
              'seasonality_mode': ['additive', 'multiplicative']
              }

# %%
all_params_m = [dict(zip(param_grid_m.keys(), value))
              for value in itertools.product(*param_grid_m.values())]
rmse_values_m = []

#%%
"""
After defining the hyperparameter search space, the model 
is trained using all permutations of hyperparameters specified 
and afterwards evaluated. This is implemented as follows:
"""

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
"""
Several metrics are available for evaluating the forecasting model.
For this task, RMSE was used. The best parameter combination are the 
ones that produces the lowest RMSE and this is identified as follows.

"""
#%%

best_params_m = all_params_m[np.argmin(rmse_values_m)]
print(best_params_m)

#%%
"""
After identifying the best parameters from the hyperparameter optimization,
they are used to fit the model, cross evaluated and their results plotted.
Also, it is worthy to note that the hyperparameter optimization selected 
the best seasonality mode to be multiplicative which is in line with the 
initial insight from the exploratry analysis that trend was multiplicative.
"""

#%% build model with best parameters

best_params_model_m = Prophet(**best_params_m)
best_params_model_m.add_regressor(name='view_duration')
best_params_model_m.fit(brochure_install_prep)

#%%

best_params_cv_res = cross_validation(best_params_model_m, 
                                            horizon='30 days'#, parallel='processes'
                                        )

#%%
best_param_perfmetric = performance_metrics(best_params_cv_res, rolling_window=1)

#%%
best_param_perfmetric

"""
The result for model with best parameters from optimization 
is a RMSE of 343.078616. This shows an improvement of performance of model 
with default parameters. 
"""
#%%

plot_cross_validation_metric(best_params_cv_res, metric='rmse')


#%% Using deep learning for time series forecasting

"""
Several approaches exist for time series forecasting and deep learning has 
proven to be one of the accurate ones hence explored here. Recurrent Neural 
Networks (RNN) are designed to be well suited for sequence data and 
Long Short Term Memory (LSTM) has made significant contribution in 
this area. A number of deep learning models will be 
experimented. Due to the limited time for computation, the models will 
be trained with limited epoch and an architecture that allows experimentation within 
the timeframe allowed.


Features used

In addition to the time series data, date and time related features were generated 
from the datetime feature for the analysis. This included, day of the week and month,
whether a given day is a weekend or not, hour of the day among others.

Horiron style is adopted for training and evaluation the forecast. 
The models are trained on a 48 hours window and then used to predict 
the next 10 hours horizon. By this, when the model is trained on 
1 to 48 hours, it is used to forecast 49th to 50th hour and then the 
trained on 2nd to 49th hour data to predict the 50th hour to 60th hour
in that manner. This informed how the data was splitted and 
prepared for training. For the testing dataset, the last 10 hours of 
all the was reserved. The remaining dataset is used for 
training and validation.

"""
#%%

"""
## Feature engineering
The datetime features were created as follows:
"""

#%%
brochure_install_prep_transform = brochure_install_prep.copy()

brochure_install_prep_transform['day_of_week'] = brochure_install_prep_transform['ds'].dt.day_of_week
brochure_install_prep_transform['hour'] = brochure_install_prep_transform['ds'].dt.hour
brochure_install_prep_transform['month_day'] = brochure_install_prep_transform['ds'].dt.day
brochure_install_prep_transform['weekend'] = (np.where(brochure_install_prep_transform['ds']
                                             .dt.day_name().isin(["Saturday", "Sunday"]), 
                                             1, 0
                                             )
                                    )


#%% # take last 10 hour as testing dataset

validate_multivar = brochure_install_prep_transform.tail(10)
#brochure_install_prep_transform[2:].tail(10)

#%%
# The rest of the data excluding testing dataset is used for training and validation.
train_multivar = brochure_install_prep_transform.drop(validate_multivar.index)


#%%
"""
### Data preprocessing

Numeric features such as day of the month, view duration among others 
were scaled using MinMax scaling. The weekend feature which indicates,
whether a day is a weekend or not is binary with weekends assigned the 
value of 1. Thus, this feature is not numeric hence not scaled like 
the other numeric features.
Normalization is done as an attempt to reduce training time for 
convergence  
"""
#%%


x_multi_scaler = preprocessing.MinMaxScaler()

#%%
columns_to_scale = train_multivar.columns[2:-1]

#%%

x_multi_scaler.fit_transform(train_multivar[columns_to_scale]).shape

#%%
train_multivar[columns_to_scale] = x_multi_scaler.transform(train_multivar[columns_to_scale])



#%%

"""
### Data splitting for horion style forecasting

Splitting the data for horizon style forecasting as describe earlier is 

undertake as follows:
"""

#%%
def horizon_style_data_splitter(predictors: pd.DataFrame, 
                                target: pd.DataFrame, start: int, 
                                end: int, window: int, horizon: int
                                ):
    X = []
    y = []
    start = start + window
    if end is None:
        end = len(predictors) - horizon
        
    for i in range(start, end):
        indices = range(i-window, i)
        X.append(predictors.iloc[indices])
        
        indicey = range(i+1, i+1+horizon)
        y.append(target.iloc[indicey])
    return np.array(X), np.array(y)


dataX = train_multivar[train_multivar.columns[1:]] #x_multi_scaler.transform(train_multivar[columns_to_scale]) #

dataY = train_multivar[['y']] #y_scaler.fit_transform(train_multivar[['y']]) #

#%%  ##########  

hist_window_multi = 48
horizon_multi = 10
TRAIN_SPLIT = 2100

x_train_multi, y_train_multi = horizon_style_data_splitter(predictors=dataX, target=dataY,
                                                         start=0, end=TRAIN_SPLIT,
                                                         window=hist_window_multi,
                                                         horizon=horizon_multi
                                                         )

x_val_multi, y_val_multi = horizon_style_data_splitter(predictors=dataX, target=dataY,
                                                     start=TRAIN_SPLIT, end=None,
                                                     window=hist_window_multi,
                                                     horizon=horizon_multi
                                                     )


#%%

"""
During each training session, only a subset of the all training samples are used. 
Training in batches requires specifing the sample to to train for each mini-batch.
This is define to be 64.
"""

#%%

BATCH_SIZE = 64
BUFFER_SIZE = 100


train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

#%%

"""
### Evaluation of models

The performance of the model will be assessed based on RMSE and other evaluation metrics 
will be reported. This will reported using function defined below:
"""

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
 

#%% plot model loss history
"""
Understanding the validation loss changes over various epoch is important to gauge whether 
further training will bring about any improvement to decide the next line of action. 
For this, a function is defined to plot the loss history as follows:

"""

def plot_loss_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'validation loss'], loc='upper left')
    plt.rcParams['figure.figsize'] = [16, 9]
    return plt.show()



#%%

"""
Long Short Term Memory (LSTM) for forecasting

LSTM is define for training and forecasting page turn count. The architecture is defined 
to include 556 neurons for the input layer that feeds directly to a hidden layer 
of 556 neurons and two other hidden layers down stream with 256 neurons. The output layer
makes a prediction for 10 outputs which corresponds to the number of forecasting horizons
defined to be 10 hours. This simplified architecture forms the basis for most of the 
models implemented. LSTM is defined as follows:

"""

#%%
tf.random.set_seed(2023)
np.random.seed(2023)

lstm_multi = tf.keras.models.Sequential()
lstm_multi.add(tf.keras.layers.LSTM(units=556, input_shape=x_train_multi.shape[-2:], 
                                    return_sequences=True)
               )
#lstm_multi.add(tf.keras.layers.Dropout(0.2))
lstm_multi.add(tf.keras.layers.LSTM(units=556, return_sequences=False))
#lstm_multi.add(tf.keras.layers.Dropout(0.2))
lstm_multi.add(tf.keras.layers.Dense(256))
lstm_multi.add(tf.keras.layers.Dense(256))
lstm_multi.add(tf.keras.layers.Dense(units=horizon_multi))
lstm_multi.compile(optimizer='adam', loss='mse')

#%%
model_path_lstm_multi = 'lstm_multivariate.h5' 


#%%
EVALUATION_INTERVAL = 20
EPOCHS = 10
history_multi = lstm_multi.fit(x=train_data_multi, epochs=EPOCHS, 
                         steps_per_epoch=EVALUATION_INTERVAL,
                         validation_data=val_data_multi,
                         validation_steps=5, verbose=1, 
                         callbacks=[
                            #  tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                            #                                          min_delta=0,patience=10,
                            #                                          verbose=1,mode='min'
                            #                                          ),
                                    tf.keras.callbacks.ModelCheckpoint(model_path_lstm_multi, 
                                                                       monitor='val_loss',
                                                                       save_best_only=True,
                                                                       mode="min",
                                                                       verbose=0
                                                                       )
                                    ]
                         )



#%%
trained_model_multi = tf.keras.models.load_model(model_path_lstm_multi)

plot_loss_history(history_multi)

#%% ## show model architechture

trained_model_multi.summary()

#%% take the last 48 hours as input for prediction

"""
### Preparing for forecasting

To forecast the number of page turn count for next 10 hours, 
the sequence of values as well as datetime features extracted are used. 
For this, the last 48 hours are feed to the model to forecast for the next 10 hours.
The forecast is compared with the ground truth in the test dataset and evaluation 
metrics are computed. This is implemented as follows:


"""

#%%
data_val = train_multivar[train_multivar.columns[1:]].tail(48)
val_rescaled = np.array(data_val).reshape(1, data_val.shape[0], data_val.shape[1])
predicted_results = trained_model_multi.predict(val_rescaled)

#%%
timeseries_evaluation_metrics_func(y_true=validate_multivar['y'],
                                   y_pred=predicted_results[0]
                                   )


#%%

plt.plot(list(validate_multivar['y']))
plt.plot(list(predicted_results[0]))
plt.title("Actual vs Predicted")
plt.ylabel("page turn count")
plt.legend(('Actual', 'Predicted'))
plt.show()


def compare_forecast_actual_graph(forecast, actual, title="Actual vs Predicted",
                                  yaxsis_label="page turn count",
                                  xaxsis_label="horizon (hourly)",
                                  legend: set = ('Actual', 'Predicted')):
    plt.plot(list(actual['y']))
    plt.plot(list(forecast[0]))
    plt.title(title)
    plt.ylabel(yaxsis_label)
    plt.xlabel(xaxsis_label)
    plt.legend(legend)
    return plt.show()

compare_forecast_actual_graph(forecast=validate_multivar, actual=predicted_results)



#%%
"""
The above process depicts how to use deep learning particularly LSTM for time series 
forecasting using both sequence of data and datetime features. This is a case of 
multivariant time series forecasting. 

The logic presented here holds true for a number of other deep learning forecasting models 
which are experimented hereafter. The code for such experimentation is provided for 
Bidirectional LSTM and Convolutional Neural Networks (CNN) as follows
"""
### 
    






#%%

## first the required packages are imported and data is read
from numpy.random import seed
seed(2022)
import tensorflow as tf
tf.random.set_seed(2022)
import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import matplotlib.pyplot as plt
from keras.callbacks import History  
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences


history = History()




