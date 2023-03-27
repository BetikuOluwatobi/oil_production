#!/usr/bin/env python3

import os
import sys
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.impute import SimpleImputer
import pandas as pd
import zipfile
import tensorflow as tf
from tensorflow.data import Dataset


class Preprocessing(object):

    def __init__(self):
        print('latty instantiated')
        pass

    def import_data_from_kaggle_(self):
        api = KaggleApi()
        api.authenticate()
        api.competition_download_file('datascienceatraisa',
                             'Harmony_data.csv', path='./data')
        api.competition_download_file('datascienceatraisa',
                                    'IHS_data.csv', path='./data')
        api.competition_download_file('datascienceatraisa',
                                    'production_data_train.csv', path='./data')
        files = ['IHS_data.csv.zip', 'production_data_train.csv.zip']

        for file in files:
            print(file)
            with zipfile.ZipFile(file, mode="r") as zipper:
                zipper.extractall('./data')

    def read_data_(self):
        self.ihs_data = pd.read_csv(os.path.join(os.getcwd(),'data','IHS_data.csv'))
        self.harmony_data = pd.read_csv(os.path.join(os.getcwd(),'data','Harmony_data.csv'))
        self.production_train = pd.read_csv(os.path.join(os.getcwd(),'data','production_data_train.csv'))

        return self.ihs_data,self.harmony_data,self.production_train

    def filterDuplicates_(self):
        self.ihs_data = self.ihs_data.drop_duplicates(subset=['API','FirstProductionDate'])
        self.ihs_data = self.ihs_data.dropna(how='any')
        self.ihs_data = self.ihs_data.drop_duplicates(subset='API')
        return self.ihs_data

    def get_labels_(self):
        '''
        This function creates the data label for the training set from
        the production train data and returns the first three years 
        cumulative production.
        
        Args:
        df = production_data_train
        Output:
        result = Data Frame with 3 years cumulative production for each well
        '''
        sample = self.production_train.groupby(['API','Year']).sum()
        
        result = []
        for idx in set(sample.index.get_level_values(0)):
            cum_production = sample.loc[idx].iloc[:3]['Liquid'].sum()
            result.append([idx,cum_production])

        return pd.DataFrame(data=result,columns=['API', '3_yrs_cum_oil'])

    def preprocess_(self):
        self.ihs_data = self.filterDuplicates_()
        drop_col = ['FirstProductionDate','CompletionDate','SpudDate','PermitDate','operatorNameIHS','formation','BasinName','StateName','CountyName']
        self.ihs_final = self.ihs_data.drop(drop_col,axis=1)
        self.harmony_data['PROP_PER_FOOT'] = self.harmony_data['PROP_PER_FOOT'].interpolate(method ='linear', limit_direction ='forward')
        self.harmony_data['WATER_PER_FOOT'] = self.harmony_data['WATER_PER_FOOT'].interpolate(method ='linear', limit_direction ='forward')
        sm = SimpleImputer(strategy='mean')
        self.harmony_data[['GOR_30','GOR_60','GOR_90']] = sm.fit_transform(self.harmony_data[['GOR_30','GOR_60','GOR_90']])
        self.production_labels = self.get_labels_()
        self.df = pd.merge(left=self.harmony_data, right=self.ihs_final, how='inner',on='API').merge(right=self.production_labels,how='inner',on='API')
        self.df = self.df.set_index('API')
        
        return self.df

    def preprocess_dataset(self,X_train,X_test,y_train,y_test):
        batch_size = 64
        buffer_size = 1000
        size = X_train.shape[-1]

        dataset = Dataset.from_tensor_slices((X_train,y_train))
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size=batch_size,drop_remainder=True).prefetch(1)


        test_dataset = Dataset.from_tensor_slices((X_test,y_test))
        test_dataset = test_dataset.shuffle(buffer_size=buffer_size)
        test_dataset = test_dataset.batch(batch_size=batch_size,drop_remainder=True).prefetch(1)

        return dataset, test_dataset
