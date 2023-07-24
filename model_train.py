#%%
from preprocess_pipeline import PipelineBuilder
import pandas as pd 
from joblib import dump, load
from sklearn.metrics import mean_squared_error
from arguments import args
from candidate_models import candidate_regressors
from typing import List, Tuple
import plotly.express as px




pipeline = PipelineBuilder()

model_pipeline = pipeline.build_model_pipeline()

class Model(object):
    def __init__(self, training_features: pd.DataFrame, 
                 training_target_variable: pd.DataFrame,
                 test_features: pd.DataFrame = None, 
                 test_target_variable: pd.DataFrame = None
                 ):
        self.training_features = training_features
        self.training_target_variable = training_target_variable
        self.test_features = test_features
        self.test_target_variable = test_target_variable
    
    #@classmethod
    def model_fit(self, model_pipeline = model_pipeline):
        self.model_pipeline = model_pipeline
        
        self.model_fitted = self.model_pipeline.fit(self.training_features, 
                                                    self.training_target_variable
                                                    )
        return self.model_fitted
        
    
    #@classmethod
    def predict_values(self, test_features = None):
        if test_features is not None:
            self.test_features = test_features
        self.predictions = self.model_fitted.predict(self.test_features)
        return self.predictions
    
    #@property
    def evaluate_model(self, y_true = None, y_pred = None):
        if y_true is None:
            y_true = self.test_target_variable
        else:
            y_true = y_true
            
        if y_pred is None:
            y_pred = self.predictions
        else:
            y_pred = y_pred
        
        self.model_rmse = mean_squared_error(y_true=y_true, 
                                            y_pred=y_pred,
                                            squared=False
                                            )
        return self.model_rmse
    
    
    def save_model(self, model = None, filename = args.model_store_path):
        if model is not None:
            model_to_save = model
        model_to_save = self.model_fitted
        dump(value=model_to_save, filename=filename)
        print('model successfully saved')
        
        
    def run_regressors(self, estimators: dict = candidate_regressors):
        test_rmse_list = []
        model_name_list = []
        
        for model_name, mod_pipeline in estimators.items(): 
            model_name_list.append(model_name)  
            
            self.mod_fitted = self.model_fit(model_pipeline=mod_pipeline)
            predicted_values = self.predict_values()
            rmse = self.evaluate_model(y_pred=predicted_values) 
            test_rmse_list.append(rmse) 
            
        regressors_eval_list = {'model_name': model_name_list, 
                            'test_rmse': test_rmse_list
                            }
        self.models_performance = pd.DataFrame(regressors_eval_list)
        return self.models_performance              
          
        
    def plot_regressors_test_rmse_results(self):
        self.test_rmse_fig = px.bar(data_frame=self.models_performance, 
                                x='model_name', y='test_rmse', 
                                color='model_name',
                                title=f'Test RMSE of various models assessed',
                                template='plotly_dark', height=700,
                                )
        return self.test_rmse_fig

        
    def get_best_model_name(self, models_test_result_df = None, 
                            colname_for_models: str = 'model_name', 
                   colname_for_score: str = 'test_rmse') -> str:
            """Accepts data for models and test score and returns the name of best model detected after
                from running all candidate models

            Args:
                models_fit_df (pd.DataFrame): Dataframe with columns for model name and test score
                colname_for_models (str): Name of column with values as model names
                colname_for_score (str): Name of column with values as test score (RMSE)
                
            Returns: 
                The name of the best algorithm (model)
            """
            if models_test_result_df is None:
                self.models_test_result_df = self.models_performance
            else:
                self.models_test_result_df = models_test_result_df
            
            max_acc = self.models_test_result_df[colname_for_score].min()
            best_model_name = (self.models_test_result_df[self.models_test_result_df
                                                          [colname_for_score]==max_acc
                                                          ]
                               [colname_for_models].item()
                               )
            return best_model_name
   

    @property
    def best_model_fitted(self, candidate_models = candidate_regressors):
        """Retrieves best candidate model pipeline and fit on data
        
        Returns:
            Model pipeline

        """
        best_model_name = self.get_best_model_name()
        
        best_model_pipeline = candidate_models[best_model_name]
        print("best candidate model being fit on data")
        _best_model_fitted = self.model_fit(model_pipeline=best_model_pipeline)
        print("model fitting completed")
        return _best_model_fitted
    
    def save_best_model(self):
        self.save_model(model=self.best_model_fitted, 
                        filename=args.best_model_store_path
                        )


            
                
        


  
