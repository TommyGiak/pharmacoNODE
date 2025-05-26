import models
import utils

import sys
import os

import pandas as pd
import configparser

import sys

PATIENTS = sys.argv[1:]
assert len(PATIENTS) > 0, print('Insert at least one patient ID')

# working directory
wd = os.path.dirname(__file__) 
config = configparser.ConfigParser()
config.read(os.path.join(wd,'config.ini'))

#import data from file
data_file = os.path.join(wd, config['PATHS']['path_data'], config['NAMES']['data_file_predict'])
data_csv = pd.read_csv(data_file, index_col=config['NAMES']['ID_col'])

# model path from config file
path_model = os.path.join(wd, config['PATHS']['path_models'], config['NAMES']['model_name'])

#check if the column names are correct
assert utils.check_col_names(data_csv.columns, config['NAMES']['col_names'].split(','))

#get the covariates columns
covariates = utils.get_covariates_col(data_csv.columns, config['NAMES']['col_names'].split(','))

# create the data class
data = utils.create_data_class(data_csv, config['NAMES']['col_names'].split(','), covariates)
data.convert_to_tensor()

# load the model
node = utils.load_NODE(path_model)

# plot the patient
for p in PATIENTS:
    node.plot_patient(data,p)


