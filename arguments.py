from argparse import Namespace


args = Namespace(
    target_variable = 'y',                   
    categorical_features = [],
    selected_predictors = ['view_duration', 'day_of_week', 'hour', 'month_day', 'weekend'],
    selected_numeric_features = ['view_duration', 'day_of_week', 'hour', 'month_day'],
    model_store_path = 'model_store/model.model',
    best_model_store_path = 'model_store/best_model.model'

)

	