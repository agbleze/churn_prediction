#%%

from xgboost import  XGBRegressor, XGBRFRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor
                              #HistGradientBoostingRegressor,
                              #)
from sklearn.ensemble import HistGradientBoostingRegressor
#from sklearn.neighbors import KNeighborsClassifier
from preprocess_pipeline import PipelineBuilder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from typing import List, Tuple
from sklearn.model_selection import cross_val_score,cross_validate
import pandas as pd


pipeline = PipelineBuilder()

preprocess_pipeline = pipeline.build_data_preprocess_pipeline()

linear = LinearRegression()

svr_rbf = SVR(kernel='rbf')
svr_linear = SVR(kernel='linear')
svr_poly = SVR(kernel='poly')
rfr = RandomForestRegressor()

xgb = XGBRFRegressor()
xgbrf = XGBRFRegressor()

hgb = HistGradientBoostingRegressor()

decision_tree = DecisionTreeRegressor()
extra_decision_tree = ExtraTreeRegressor()

linear_pipeline = pipeline.build_model_pipeline(model=linear)
svr_rbf_pipeline = pipeline.build_model_pipeline(model=svr_rbf)
svr_linear_pipeline = pipeline.build_model_pipeline(model=svr_linear)
svr_poly_pipeline = pipeline.build_model_pipeline(model=svr_poly)
rfr_pipeline = pipeline.build_model_pipeline(model=rfr)
decision_tree_pipeline = pipeline.build_model_pipeline(model=decision_tree)
extra_decision_tree_pipeline = pipeline.build_model_pipeline(model=extra_decision_tree)
xgb_pipeline = pipeline.build_model_pipeline(model=xgb)
xgbrf_pipeline = pipeline.build_model_pipeline(model=xgbrf)
hgb_pipeline = pipeline.build_model_pipeline(model=hgb)

candidate_regressors = {"Extra decision tree": extra_decision_tree_pipeline,
                        "Decision tree": decision_tree_pipeline,
                        "Radom forest Regressor": rfr_pipeline,
                        "SVR poly": svr_poly_pipeline,
                        "SVR linear": svr_linear_pipeline,
                        "SVR rbf": svr_rbf_pipeline,
                        "Linear regression": linear_pipeline,
                        "Extreme Gradient Boosting": xgb_pipeline,
                        "Extreme gradient boosting with Random Forest": xgbrf_pipeline,
                        "Histogram gradient boosting": hgb_pipeline
                        }


