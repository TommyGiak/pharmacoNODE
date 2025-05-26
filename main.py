import os
import sys

import pandas as pd

import utils
import models

import configparser

wd = os.path.dirname(__file__) # working directory
config = configparser.ConfigParser()
config.read(os.path.join(wd,'config.ini'))

include_covariates = True if config['NN_SETTINGS']['include_covariates'] == 'True' else False

#import data from file
os.makedirs(os.path.join(wd, config['PATHS']['path_data']), exist_ok=True)
data_file = os.path.join(wd, config['PATHS']['path_data'], config['NAMES']['data_file'])
data_csv = pd.read_csv(data_file, index_col=config['NAMES']['ID_col'])

#check if the column names are correct
assert utils.check_col_names(data_csv.columns, config['NAMES']['col_names'].split(','))

#get the covariates columns
if include_covariates:
    covariates = utils.get_covariates_col(data_csv.columns, config['NAMES']['col_names'].split(','))
else:
    covariates = None

# create the data class
data = utils.create_data_class(data_csv, config['NAMES']['col_names'].split(','), covariates)

# fit the two compartment model and get the parameters to generate synthetic data
params = utils.get_ode_params(data)

# generate synthetic data from real data
synth_data = utils.generate_sythetic_data(params, utils.get_doses(data))


#converting to tensors
data.convert_to_tensor()
synth_data.convert_to_tensor() 

# define the NODE model
dim_c = list(map(int, config['NN_SETTINGS']['dim_c'].split(',')))
dim_V = list(map(int, config['NN_SETTINGS']['dim_V'].split(','))) if include_covariates else None
node = models.NODE(dim_c, dim_V, n_cov=len(covariates)) if include_covariates else models.NODE(dim_c)

# pre-training with synthetic data
epochs = int(config['TRAIN_SETTINGS']['epochs_synth'])
lr = float(config['TRAIN_SETTINGS']['lr'])
weights_decay = float(config['TRAIN_SETTINGS']['weights_decay'])
print(f'Statring pre-training with synthetic data for {epochs} epochs')
lossi_synth = node.train_synthetic(synth_data, data.covariates, epochs, lr, 
weights_decay)
print(f'Loss after pre-training with synthetic data: {lossi_synth[-1]:.4f}\n')

# training with real data
epochs = int(config['TRAIN_SETTINGS']['epochs_train'])
print(f'Statring training with real data for {epochs} epochs')
lossi_real = node.train(data, epochs, lr, weights_decay)
print(f'Loss after training with real data: {lossi_real[-1]:.4f}\n')

#fine tuning with real data
epochs = int(config['TRAIN_SETTINGS']['epochs_fine_tuning'])
lr = float(config['TRAIN_SETTINGS']['lr_reduced'])
print(f'Statring fine tuning with real data for {epochs} epochs')
lossi_fine = node.train(data, epochs, lr, weights_decay)
print(f'Loss after fine tuning with real data: {lossi_fine[-1]:.4f}\n')

# Save the NODE model
os.makedirs(os.path.join(wd, config['PATHS']['path_models']), exist_ok=True)
model_path = os.path.join(wd, config['PATHS']['path_models'], config['NAMES']['model_name'])
if not os.path.exists(model_path):
    os.makedirs(model_path)
node.save(model_path)
print(f'Model saved to {model_path}\n')