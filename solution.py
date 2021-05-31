import autogluon as ag
from autogluon.tabular import TabularPrediction as task
import pandas as pd
import numpy as np
import os
import urllib.request

def main() -> None:
    # download data
    if not os.path.isfile('train.csv'):
        urllib.request.urlretrieve('https://datahack-prod.s3.amazonaws.com/train_file/train_s3TEQDk.csv ', 'train.csv')
    if not os.path.isfile('test.csv'):
        urllib.request.urlretrieve('https://datahack-prod.s3.amazonaws.com/test_file/test_mSzZ8RL.csv', 'test.csv')

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    test_id = test.ID
    test.drop('ID', axis=1, inplace=True)
    train.drop('ID', axis=1, inplace=True)
    train = train.sample(frac=1, random_state=25).reset_index(drop=True)

    hyperparameters = {  # hyperparameters of each model type
                      'GBM': [
                      {'extra_trees': True},
                      {},
                      ],
                      'CAT': {},
                      'XGB': {'nthread': -1},
                      'RF':
                      {'criterion': 'gini', 'n_jobs': -1, 'max_depth': 10},
                      'XT': [
                      {'criterion': 'gini', 'n_jobs': -1, 'max_depth': 10},
                      {'criterion': 'entropy', 'n_jobs': -1, 'max_depth': 10},
                      ],
                      'KNN': [
                      {'weights': 'uniform', 'n_jobs': -1},
                      {'weights': 'distance','n_jobs': -1,},
                      ],
                  }
 
    time_limits = 60*60*2 #seconds
    metric = 'roc_auc'
    predictor = task.fit(train_data=train, label='Is_Lead', time_limits=time_limits, presets='optimize_for_deployment', hyperparameters = hyperparameters,
                        eval_metric=metric, num_bagging_folds=5, stack_ensemble_levels=3,num_bagging_sets=1,output_directory="saved_models")

    predictions = predictor.predict_proba(test)
    solution = test_id.to_frame()
    solution['Is_Lead'] = predictions
    solution.to_csv('jobathon_3_stack_seed_25.csv',index=False)

if __name__ == "__main__":
    main()