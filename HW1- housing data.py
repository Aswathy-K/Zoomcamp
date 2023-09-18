# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 09:11:45 2023

@author: a8.kumar
"""

import pandas as pd
import numpy as np
house_data = pd.read_csv('datasets_master_housing.csv')
print(house_data.head())

sahpe_house_data = house_data.shape

miss_house_data = house_data.isna().sum()

no_val_ocean_prox = len(pd.unique(house_data['ocean_proximity']))

#select near bay houses
ocean_house = house_data[house_data['ocean_proximity']=='NEAR BAY']
avg_house_val = np.mean(ocean_house['median_house_value'])

np.mean(house_data['total_bedrooms'])
#replace NA with mean
house_data['total_bedrooms'].fillna(house_data['total_bedrooms'].mean(), inplace= True)

house_data['total_bedrooms'].mean()

island_data = house_data[house_data['ocean_proximity']=='ISLAND']
X = island_data[['housing_median_age','total_rooms','total_bedrooms']]

XT = X.transpose()

XTX = np.dot(XT,X)

y = [950,1300,800,1000,1300]

i = np.linalg.inv(XTX)

w=np.dot(i,XT)

w= np.dot(w,y)