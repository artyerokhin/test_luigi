import pandas as pd
import numpy as np

from geopy.distance import vincenty

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from dateutil.parser import parse

import requests
import json
import os
import datetime
import pickle

from tqdm import tqdm

from sqlalchemy import create_engine
import sqlite3

# must have this config.py file with config={'secret':...} structure
from config import config

def create_paths(path_list):
    """
        create paths from [path_list] in directory
    """
    for path in path_list:
        os.makedirs(path, exist_ok=True)

def date_hour():
    """
        return date_hour from datetime.now
    """

    dt = datetime.datetime.now()

    return '{}_{}'.format(dt.date(), dt.hour)

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

def load_nodes(file_path='krasnodar_weights_full.csv'):
    """
        load nodes from file
    """

    df = pd.read_csv(file_path)

    x = df[['source_id','source_coordinate']].rename(columns={'source_id':'node_id',
            'source_coordinate':'node_coordinate'}).append(df[['destination_id','destination_coordinate']].rename(
                columns={'destination_id':'node_id', 'destination_coordinate':'node_coordinate'})).drop_duplicates(
                                                                                                    subset='node_id')

    x['lat'] = [float(i.replace('(','').replace(')','').split(',')[0]) for i in x.node_coordinate]
    x['lon'] = [float(i.replace('(','').replace(')','').split(',')[1]) for i in x.node_coordinate]

    return x

def make_df(nodes_df, config, n_points=10):
    """
        make dataframe with uber API requests results
    """

    json_array = []
    results_array = []
    datetimes = []

    rn = pd.DataFrame(random_nodes(nodes_df[['node_id','lat','lon']], n_points=n_points),
                  columns=['start_node_id','start_lat','start_lon','end_node_id','end_lat','end_lon'])

    rn.start_node_id = rn.start_node_id.astype(int)
    rn.end_node_id = rn.end_node_id.astype(int)

    for val in tqdm(rn[['start_lat','start_lon', 'end_lat','end_lon']].itertuples(index=False)):

        j = api_request((val[0], val[1]), (val[2], val[3]), secret=config['secret'])
        parsed_data = parse_json(j, length='m')

        json_array.append(str(j))
        results_array.append(parsed_data)
        datetimes.append(datetime.datetime.now())

    results_array = np.array(results_array)

    rn['datetime'] = datetimes
    rn['price'] = results_array[:,0]
    rn['distance'] = results_array[:,1]
    rn['time'] = results_array[:,2]
    rn['json'] = json_array

    return rn[['datetime', 'start_lat','start_lon', 'end_lat','end_lon', 'price', 'distance', 'time', 'json']]

def execute_sql(sql_query, database):
    """
        execute sql query
    """

    conn = sqlite3.connect('{}'.format(database))
    cursrsor = conn.cursor()

    # we can't use construction with conn.cursor() as cursor in sqlite3
    cursor.execute(sql_query)
    conn.close()

def check_table_existance(database, table, params):
    """
        check table existance in database
        create table if not exists
    """

    conn = sqlite3.connect('{}'.format(database))
    cursor = conn.cursor()

    # we can't use construction with conn.cursor() as cursor in sqlite3
    cursor.execute('CREATE TABLE IF NOT EXISTS {} ({});'.format(table, ','.join(params)))
    conn.close()

def load_max_row(database, table, id_column='row_id'):
    """
        load maximum row_id number
    """
    # connection with alchemy
    disk_engine = create_engine('sqlite:///{}'.format(database))
    # save with to_sql pandas method
    pd.read_sql_query('SELECT MAX({}) FROM {}'.format(id_column, table), disk_engine)

def insert_values(database, table, values):
    """
        insert [values] into [database].[table]
    """

    sql_query = "INSERT INTO {} VALUES ({})".format(table, values)

    execute_sql(sql_query, database)

def insert_df_values(df, database, table):
    """
        insert values from [df] into [database].[table]
    """

    # connection with alchemy
    disk_engine = create_engine('sqlite:///{}'.format(database))
    # save with to_sql pandas method
    df.to_sql('{}'.format(table), disk_engine, if_exists='append', index=False)

def load_dataset(file_path, datetime_column='datetime',
        return_cols=['row_id','day','year','hour','weekday','is_workday',
                                'start_lat','start_lon','end_lat','end_lon']):
    """
        load dataset with features into dataframe
    """

    df = pd.read_csv(file_path, parse_dates=['datetime'])

    df['day'] = df.datetime.dt.day
    df['year'] = df.datetime.dt.year
    df['hour'] = df.datetime.dt.hour
    df['weekday'] = df.datetime.dt.weekday
    df['is_workday'] = df.datetime.dt.weekday//5

    return df[return_cols], df['time']

def last_model(models_path=None):
    """
        return last model path from [models_path]
    """

    if models_path is None:
        return ''
    elif len(os.listdir(models_path)) == 0:
        return ''
    else:
        return os.path.join(models_path, np.sort(os.listdir(models_path))[-1])

def normalaize_features(features):
    """
        normalaize features
    """

    ss = StandardScaler()

    return ss.fit_transform(features)

def fit_model(features, y):
    """
        create initial model
    """

    sgd = SGDRegressor()

    sgd.fit(features, y)

    return sgd

def update_model(model, features, y):
    """
        partial fit of model
    """

    model.partial_fit(features, y)

    return model

def kfold_train(data, y, model_path=''):
    """
        train model with kfold cross-validation
    """

    features = normalaize_features(data)
    kf = KFold(n_splits=5)

    for train_index, test_index in kf.split(features):
        if os.path.isfile(model_path):
            with open(model_path, 'rb') as f:
                data_new = pickle.load(f)
            model = update_model(features[train_index], y[train_index])
        else:
            model = fit_model(features[train_index], y[train_index])

        data.loc[test_index, 'prediction'] = model.predict(features[test_index])

    return data

def make_model(data, y, model_path=''):
    """
        train pkl model or make new
    """

    features = normalaize_features(data)
    if os.path.isfile(model_path):
        with open(model_path, 'rb') as f:
            data_new = pickle.load(f)
        model = update_model(features, y)
    else:
        model = fit_model(features, y)

    return model
