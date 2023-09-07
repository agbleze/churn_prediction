
#%%
# import all models to be used

from get_path import get_data_path
from datetime_feature_transform import DatetimeFeatureTransformer
from model_train import Model
import pandas as pd
import os
from parser_getter import get_args
import logging



if __name__ == '__main__':
    args = get_args()
    logging.info("argument successfully retrieved")
    datapath = get_data_path(folder_name=args.data_store, file_name=args.file_name)
    data = pd.read_csv(datapath)

    # The class for extracted date and time related features is implemented as follows
    dft = DatetimeFeatureTransformer(data=data)

    dft.datetime_feature_transform(datetime_column='ds')
    ## A copy of the data is made and prepared for splitting
    transformed_data = dft.data.copy()
    transformed_data.drop(columns=['Unnamed: 0'], inplace=True)
    train_ml_df = transformed_data[transformed_data['ds'].dt.month < 7].copy()

    test_ml_df = transformed_data[transformed_data['ds'].dt.month > 6]

    # The predictors and target variable are defined as follows
    X_train = train_ml_df[['view_duration', 'day_of_week', 'hour', 'month_day', 'weekend']]

    y_train = train_ml_df[['y']]

    X_test = test_ml_df[['view_duration', 'day_of_week', 'hour', 'month_day', 'weekend']]

    y_test = test_ml_df[['y']]

    # Class for the model is initiatlized using the dataset as follows:
    reg_model = Model(training_features=X_train, training_target_variable=y_train,
                        test_features=X_test, test_target_variable=y_test
                        )


    model_res = reg_model.run_regressors()
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    model_res.to_csv(f"{args.save_dir}/train_model_results.csv")
    

