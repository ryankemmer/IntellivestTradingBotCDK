import datetime
from datetime import datetime
from catboost import CatBoostClassifier, CatBoostRegressor
import os
import re
import pandas as pd
import boto3

class modelObject:

    def __init__(self):
        self.s3 = boto3.client('s3')
        self.load_most_recent_models()

    def parse_to_datetime(self, date_time):
        
        #Remove unecessary "/" from string
        date_time = date_time[:-1]

        format = "%Y-%m-%d %H:%M:%S"  # The format
        datetime_str = datetime.strptime(date_time, format)
    
        return datetime_str

    def find_most_recent(self):
        most_recent = datetime.min
        most_recent_dir = None
        prefixes = self.s3.list_objects_v2(Bucket='artifactsbucketintellivest', Delimiter='/')['CommonPrefixes']
        for pref in prefixes:
            dir_name = pref['Prefix']
            print(dir_name)
            match = re.search(r'\d{4}-\d{2}-\d{2}\s', dir_name)
            if match:
                file_date_foramt = self.parse_to_datetime(dir_name)
                if file_date_foramt > most_recent:
                    most_recent = file_date_foramt
                    most_recent_dir = dir_name
        return most_recent_dir

    def load_most_recent_models(self):

        most_recent_dir = self.find_most_recent()
        print(most_recent_dir)
        if most_recent_dir == None:
            raise FileNotFoundError('Could not find most recent model')

        resp = self.s3.list_objects_v2(Bucket='artifactsbucketintellivest', Prefix=most_recent_dir)

        side_model_0 = CatBoostClassifier()
        side_model_1 = CatBoostClassifier()
        side_model_2 = CatBoostClassifier()
        side_model_3 = CatBoostClassifier()
        side_model_4 = CatBoostClassifier()
        size_model_0 = CatBoostRegressor()
        size_model_1 = CatBoostRegressor()
        size_model_2 = CatBoostRegressor()
        size_model_3 = CatBoostRegressor()
        size_model_4 = CatBoostRegressor()

        filepath= '/tmp/model'
        for obj in resp['Contents']:
            key = obj['Key']
            print(key)
            if key.startswith(most_recent_dir + 'side-0'):
                self.s3.download_file('artifactsbucketintellivest', key, filepath)
                self.side_model_0 = side_model_0.load_model(filepath)
                print('Loaded Side Model 0')
            if key.startswith(most_recent_dir + 'side-1'):
                self.s3.download_file('artifactsbucketintellivest', key, filepath)
                self.side_model_1 = side_model_1.load_model(filepath)
                print('Loaded Side Model 1')
            if key.startswith(most_recent_dir + 'side-2'):
                self.s3.download_file('artifactsbucketintellivest', key, filepath)
                self.side_model_2 = side_model_2.load_model(filepath)
                print('Loaded Side Model 2')
            if key.startswith(most_recent_dir + 'side-3'):
                self.s3.download_file('artifactsbucketintellivest', key, filepath)
                self.side_model_3 = side_model_3.load_model(filepath)
                print('Loaded Side Model 3')
            if key.startswith(most_recent_dir + 'side-4'):
                self.s3.download_file('artifactsbucketintellivest', key, filepath)
                self.side_model_4 = side_model_4.load_model(filepath)
                print('Loaded Side Model 4')
            if key.startswith(most_recent_dir + 'size-0'):
                self.s3.download_file('artifactsbucketintellivest', key, filepath)
                self.size_model_0 = size_model_0.load_model(filepath)
                print('Loaded Size Model 0')
            if key.startswith(most_recent_dir + 'size-1'):
                self.s3.download_file('artifactsbucketintellivest', key, filepath)
                self.size_model_1 = size_model_1.load_model(filepath)
                print('Loaded Size Model 1')
            if key.startswith(most_recent_dir + 'size-2'):
                self.s3.download_file('artifactsbucketintellivest', key, filepath)
                self.size_model_2 = size_model_2.load_model(filepath)
                print('Loaded Size Model 2')
            if key.startswith(most_recent_dir + 'size-3'):
                self.s3.download_file('artifactsbucketintellivest', key, filepath)
                self.size_model_3 = size_model_3.load_model(filepath)
                print('Loaded Size Model 3')
            if key.startswith(most_recent_dir + 'size-4'):
                self.s3.download_file('artifactsbucketintellivest', key, filepath)
                self.size_model_4 = size_model_4.load_model(filepath)
                print('Loaded Size Model 4')
        
    def predict_side(self, data):

        probabilities = pd.DataFrame()

        probabilities['model' + str(0) + 'profit_probability'] = self.side_model_0.predict_proba(data.loc[:, ~data.columns.str.startswith('label')])[:,1]
        probabilities['model' + str(1) + 'profit_probability'] = self.side_model_1.predict_proba(data.loc[:, ~data.columns.str.startswith('label')])[:,1]
        probabilities['model' + str(2) + 'profit_probability'] = self.side_model_2.predict_proba(data.loc[:, ~data.columns.str.startswith('label')])[:,1]
        probabilities['model' + str(3) + 'profit_probability'] = self.side_model_3.predict_proba(data.loc[:, ~data.columns.str.startswith('label')])[:,1]
        probabilities['model' + str(4) + 'profit_probability'] = self.side_model_4.predict_proba(data.loc[:, ~data.columns.str.startswith('label')])[:,1]

        probabilities['avg'] = probabilities[['model0profit_probability', 'model1profit_probability', 'model2profit_probability', 'model3profit_probability', 'model4profit_probability']].mean(axis=1)

        average_pred = probabilities['avg'].to_numpy()

        return average_pred

    def predict_size(self, data):

        probabilities = pd.DataFrame()

        probabilities['model' + str(0) + 'profit_probability'] = self.size_model_0.predict(data.loc[:, ~data.columns.str.startswith('label')])
        probabilities['model' + str(1) + 'profit_probability'] = self.size_model_1.predict(data.loc[:, ~data.columns.str.startswith('label')])
        probabilities['model' + str(2) + 'profit_probability'] = self.size_model_2.predict(data.loc[:, ~data.columns.str.startswith('label')])
        probabilities['model' + str(3) + 'profit_probability'] = self.size_model_3.predict(data.loc[:, ~data.columns.str.startswith('label')])
        probabilities['model' + str(4) + 'profit_probability'] = self.size_model_4.predict(data.loc[:, ~data.columns.str.startswith('label')])

        probabilities['avg'] = probabilities[['model0profit_probability', 'model1profit_probability', 'model2profit_probability', 'model3profit_probability', 'model4profit_probability']].mean(axis=1)

        average_pred = probabilities['avg'].to_numpy()

        return average_pred