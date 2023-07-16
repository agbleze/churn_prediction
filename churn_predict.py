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
                                .dropna(subset='dateCreated')
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



#%%
#pd.read_fwf('churn_prediction/dataset/brochure views.txt', delimiter = " ")







# %%
