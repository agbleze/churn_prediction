#%% import all models required for the analysis
from arguments import args
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np
#from argparse import Namespace
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns


one = OneHotEncoder(handle_unknown='ignore')
scaler = StandardScaler()

class PipelineBuilder(object):
    def __init__(self, num_features: list = ['view_duration', 'day_of_week', 'hour', 'month_day'],
                 categorical_features: list = []): #args.categorical_features=
        self.num_features = num_features
        self.categorical_features = categorical_features
   
    
    #@classmethod
    def build_data_preprocess_pipeline(self):
        # if categorical features is empty, build a pipeline for only numeric
        if len(self.categorical_features) == 0:
            self.preprocess_pipeline =  make_column_transformer((scaler, self.num_features),
                                                                remainder='passthrough')
        elif len(self.num_features) == 0:
            self.preprocess_pipeline =  make_column_transformer((one, self.categorical_features),
                                                                remainder='passthrough')
               
        
        self.preprocess_pipeline =  make_column_transformer((scaler, self.num_features),
                                                        (one, self.categorical_features),
                                                        remainder='passthrough'
                                                      )
        
        return self.preprocess_pipeline
        
    
    #@classmethod
    def build_model_pipeline(self, model = None, preprocess_pipeline = None):
        if (model == None):
            self.model = RandomForestRegressor()
        else:
            self.model = model
            
        if (preprocess_pipeline == None):
            self.preprocess_pipeline = self.build_data_preprocess_pipeline()
        else:
            self.preprocess_pipeline = preprocess_pipeline
            
        model_pipeline = make_pipeline(self.preprocess_pipeline,
                                        self.model
                                        )         
            
        return model_pipeline
        