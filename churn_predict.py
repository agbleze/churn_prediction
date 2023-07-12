#%%
import pandas as pd

#%%

app_starts = pd.read_csv('churn_prediction/dataset/app starts.txt', sep = "\t")
app_starts_july = pd.read_csv('churn_prediction/dataset/app starts july.txt', sep = "\t")
brochure_views = pd.read_csv('churn_prediction/dataset/brochure views.txt', sep = "\t")
brochure_views_july = pd.read_csv('churn_prediction/dataset/brochure views july.txt', sep = "\t")
installs_df = pd.read_csv('churn_prediction/dataset/installs.txt', sep = "\t")

#%%

"""
Installs are churn

"""



#%%
pd.read_fwf('churn_prediction/dataset/brochure views.txt', delimiter = " ")







# %%
