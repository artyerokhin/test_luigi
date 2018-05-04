
# coding: utf-8

# In[89]:


import pandas as pd
import numpy as np

from geopy.distance import vincenty

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_absolute_error

import requests
import json
import os
import datetime

import sqlite3


# In[6]:


# must have this config.py file with config={'secret':..., 'database':...} structure
from config import config


# In[65]:


# request for information
def api_request(start, end, secret, url="https://api.uber.com/v1.2/requests/estimate"):
    """
        request ubrt api for price/distance estimation (max per hour requests ~100)
        Args:
            start (tuple) - lat/lon tuple for start point
            end (tuple) - lat/lon tuple for end point
            secret (dict) - secret token dictionary
            url (str) - request url
        Returns:
            result (json) - json output of request (None if status_code != 200)
    """
    
    data = {"start_latitude": start[0],
       "start_longitude": start[1],
       "end_latitude": end[0],
       "end_longitude": end[1]}

    headers = {'Content-type': 'application/json', 
               'Accept-Language': 'en_US',
              'Authorization': 'Bearer {}'.format(config['secret'])}
    
    r = requests.post(url, data=json.dumps(data), headers=headers)
    
    if r.status_code != 200:
        return 
    else:
        return r.json()
    
# parse json file 
def parse_json(json, length='km'):
    """
        parse uber api result json
        Args:
            json (json) - result requested json
            length (str) - type of length values to return - miles, kilometers or meters (mi/km/m)
        Returns:
            price, distance_estimation, time_estimation
    """
    
    length_dict = {'mi':1, 'km':1.60934, 'm':1609.34}
    
    if json is None:
        return -1, -1, -1
    else:
        if length not in length_dict.keys():
            mult = 1
        else:
            mult = length_dict[length]
        
        distance_estimation = json['trip']['distance_estimate'] * mult
        time_estimation = json['trip']['duration_estimate']
        price = json['fare']['value']
            
        return price, distance_estimation, time_estimation


# In[46]:


def random_nodes(nodes_df, min_dist=500, n_points=10, patience=10):
    """
        generate n_points random point pairs from nodes dataframe with start-end distance >= min_dist 
        Args:
            nodes_df (pandas dataframe) - dataframe with node_id, lat, lon columns
            min_dist (int/float) - minimum start-end distance in meters
            n_points (int) - number of pairs for generating pairs
            patience (int) - number of iterations to generate needed amount of point pairs
        Returns:
            [start_lat, start_lon, end_lat, end_lon, start_node_id, end_node_id] (list) - coordinates and
                node id's list for start and end points
    """
    # result array and patience counter
    result = []
    counter = 0
    
    while (len(result) < n_points) & (counter < patience):
        vals = nodes_df.sample(n_points).values
        for n in range(len(vals)):
            for m in range(n+1, len(vals)):
                if vincenty(vals[n,[1,2]], vals[m,[1,2]]).m >= min_dist:
                    result.append(list(vals[n]) + list(vals[m]))
                    if len(result) >= n_points:
                        return result
    return result


# In[58]:


rn = pd.DataFrame(random_nodes(x[['node_id','lat','lon']]), 
                  columns=['start_node_id','start_lat','start_lon','end_node_id','end_lat','end_lon'])

rn.start_node_id = rn.start_node_id.astype(int)
rn.end_node_id = rn.end_node_id.astype(int)


# In[74]:


j1 = api_request(rn[['start_lat','start_lon']].values[2], rn[['end_lat','end_lon']].values[2], 
                 secret=config['secret'])
parse_json(j1, length='m')


# In[86]:


def execute_sql(sql_query):
    """
        execute sql query
    """
    
    conn = sqlite3.connect('{}'.format(databae))
    cursrsor = conn.cursor() 
    
    # we can't use construction with conn.cursor() as cursor in sqlite3
    cursor.execute(sql_query)    

def check_table_existance(database, table, params):
    """
        check table existance in database
        create table if not exists
    """
    
    conn = sqlite3.connect('{}'.format(database))
    cursor = conn.cursor()
    
    # we can't use construction with conn.cursor() as cursor in sqlite3
    cursor.execute('CREATE TABLE IF NOT EXISTS {} ({});'.format(table, ','.join(params)))


# In[88]:


check_table_existance('uber.sqlite', 'requests', ['datetime', 'start_lat', 'start_lon', 'end_lat', 'end_lon', 
                                                  'price', 'distance', 'time', 'json'])


# In[ ]:


query = 'INSERT INTO {} VALUES ({})'.format('requests', ','.join([datetime.datetime.now()] + ))


# In[22]:


def model(features, y):
    """
        create initial model
    """
    
    sgd = SGDRegressor()
    
    sgd.fit(features, y)
    

def update_model(model, features, y):
    """
        partial fit of model
    """
    
    model.partial_fit(features, y)
    
def predict(model, features):
    """
        predict result metrics with model
    """
    
    pass
    
def estimate_prediction(pred, y):
    pass


# In[21]:


def normalaize_features(features):
    """
        normalaize features
    """
    
    norm = Normalizer()
    
    return norm.fit_transform(features)


# In[23]:


df = pd.read_csv('krasnodar_weights_full.csv')


# In[27]:


x = df[['source_id','source_coordinate']].rename(columns={'source_id':'node_id', 'source_coordinate':'node_coordinate'}
                                            ).append(df[['destination_id','destination_coordinate']].rename(
    columns={'destination_id':'node_id', 'destination_coordinate':'node_coordinate'})).drop_duplicates(subset='node_id')


# In[32]:


x['lat'] = [float(i.replace('(','').replace(')','').split(',')[0]) for i in x.node_coordinate]
x['lon'] = [float(i.replace('(','').replace(')','').split(',')[1]) for i in x.node_coordinate]


# In[34]:


x.to_csv('nodes.csv', index=False)


# In[36]:





# In[15]:


r.status_code


# In[14]:


r.json()


# In[ ]:


curl -X POST      -H 'Authorization: Bearer '      -H 'Accept-Language: en_US'      -H 'Content-Type: application/json'      -d '{
       "start_latitude": 45.1197513,
       "start_longitude": 38.9752278,
       "end_latitude": 45.043968,
       "end_longitude": 38.946352
     }' "https://api.uber.com/v1.2/requests/estimate"

