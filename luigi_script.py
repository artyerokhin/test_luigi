import test_uber_api as tu
from config import config
import luigi
import os
import pickle
import pandas as pd

# ALl classes are ...

class GetApiData(luigi.Task):
    # parameter for n_points
    n = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget('data/{}.csv'.format(tu.date_hour()))

    def run(self):
        # create paths
        tu.create_paths(['data','log','validation','model'])
        # load nodes from file
        nodes_df = tu.load_nodes()
        # trying to load maximum row_id
        try:
            max_row_id = tu.load_max_row('uber.sqlite', 'requests', 'row_id')
            if max_row_id is None:
                max_row_id = 0
        except:
            max_row_id = 0
        # make dataframe from API resutls
        results = tu.make_df(nodes_df, config, n_points=self.n)
        results['row_id'] = [i for i in range(max_row_id, max_row_id + len(results))]
        # save as csv file
        results.to_csv('data/{}.csv'.format(tu.date_hour()), index=False)


class SaveApiToSql(luigi.Task):

    def requires(self):
        return GetApiData()

    def output(self):
        return luigi.LocalTarget('log/{}_api_sql.txt'.format(tu.date_hour()))

    def run(self):
        # check table existance
        tu.check_table_existance('uber.sqlite', 'requests', ['row_id','datetime', 'start_lat',
            'start_lon', 'end_lat', 'end_lon', 'price', 'distance', 'time', 'json'])
        # load values
        results = pd.read_csv('data/{}.csv'.format(tu.date_hour()))
        # insert values from dataframe into sql table
        tu.insert_df_values(results, 'uber.sqlite', 'requests')

        # idea from
        # https://stackoverflow.com/questions/42816889/replacing-a-table-load-function-with-a-luigi-task
        with open('log/{}_api_sql.txt'.format(tu.date_hour()), 'w') as f:
            f.write(tu.date_hour())


class ValidateModel(luigi.Task):

    def requires(self):
        return SaveApiToSql()

    def output(self):
        return luigi.LocalTarget('validation/{}.csv'.format(tu.date_hour()))

    def run(self):
        # load dataframe
        time_df, y = tu.load_dataset('data/{}.csv'.format(tu.date_hour()))
        # train our model with k-fold cv
        trained_df = tu.kfold_train(time_df, y, model_path=tu.last_model('model'))
        trained_df.to_csv('validation/{}.csv'.format(tu.date_hour()))


class SaveValidationToSql(luigi.Task):

    def requires(self):
        return ValidateModel()

    def output(self):
        return luigi.LocalTarget('log/{}_validation_sql.txt'.format(tu.date_hour()))

    def run(self):
        # check table existance
        tu.check_table_existance('uber.sqlite', 'predictions',
                                                    ['row_id', 'prediction'])
        # load values
        results = pd.read_csv('validation/{}.csv'.format(tu.date_hour()))
        # insert values from dataframe into sql table
        tu.insert_df_values(results[['row_id','prediction']], 'uber.sqlite',
                                                                'predictions')
        # idea from
        # https://stackoverflow.com/questions/42816889/replacing-a-table-load-function-with-a-luigi-task
        with open('log/{}_validation_sql.txt'.format(tu.date_hour()), 'w') as f:
            f.write(tu.date_hour())

class TrainModel(luigi.Task):

    def requires(self):
        return SaveValidationToSql()

    def output(self):
        return luigi.LocalTarget('model/{}.pkl'.format(tu.date_hour()))

    def run(self):
        # load dataframe
        time_df, y = tu.load_dataset('data/{}.csv'.format(tu.date_hour()))
        # train our model
        model = tu.make_model(time_df, y, model_path=tu.last_model('model'))
        # save as pickle file
        with open('model/{}.pkl'.format(tu.date_hour()), 'wb') as f:
            pickle.dump(model, f)

if __name__ == '__main__':
    luigi.run()
