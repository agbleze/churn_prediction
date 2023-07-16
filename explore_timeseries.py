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


#%% Categorical data exploration

"""

"""







