import numpy as np
import pandas as pd

class DatetimeFeatureTransformer(object):
    def __init__(self, data):
        self.data = data
        
    def datetime_feature_transform(self, datetime_column):
        #data_trn = self.data
        self.data[datetime_column] = pd.to_datetime(self.data[datetime_column])
        self.data['day_of_week'] = self.data[datetime_column].dt.day_of_week
        self.data['hour'] = self.data[datetime_column].dt.hour
        self.data['month_day'] = self.data[datetime_column].dt.day
        self.data['weekend'] = (np.where(self.data[datetime_column]
                                            .dt.day_name().isin(["Saturday", "Sunday"]), 
                                            1, 0
                                            )
                                        )
    
        return self.data
    
    def get_transformed_datetime_features(self):
        return self.data[['day_of_week', 'hour', 'month_day', 'weekend']]
    
    def get_transformed_features_column_names(self):
        return ['day_of_week', 'hour', 'month_day', 'weekend']
