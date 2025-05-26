import numpy as np
import time
import matplotlib.pyplot as plt

import torch
from torch import nn
import torchdiffeq

import utils
from scipy.integrate import odeint

import os
import pickle

from tqdm.auto import tqdm


class Data():
    '''
    Class to create the data for the csv file
    '''

    def __init__(self, data_csv, col_names, covariates):
        self.data_csv = data_csv # save the data as a pandas dataframe
        self.patients = list(set(data_csv.index)) # list of patients
        self.n_pat = len(self.patients) # number of patients
        self.cov_names = covariates # list of covariates names
        self.col_names = col_names
        self.time_name = col_names[0]
        self.dose_name = col_names[2]
        self.conc_name = col_names[3]
        self.evid_name = col_names[4]

    def create_data(self):
        '''
        Create the data for each patient as a dictionaries for concentrations, administrations and covariates
        '''
        self.conc = {}
        self.admin = {}
        self.covariates = {}
        self.time_infusion = {}
        # loop over the patients
        for p in self.patients:
            # get the data for the patient
            pat_data = self.data_csv.loc[p]
            pat_data = pat_data.sort_values(by=self.time_name) # sort the data by time
            
            measures = pat_data[pat_data[self.evid_name]==0] # get the concentration measures
            administrations = pat_data[pat_data[self.evid_name]==1] # get the administrations

            adm_count = 1
            if len(administrations) > 1: # if there are more than one administration
                self.conc[p] = np.array([0.,0.,0.]) # initialize the concentration array
                
                # loop over the administrations for each patient
                for i in range(len(administrations)-1):
                    # setting the first column of concentration measures as the number of previous administrations
                    time_previous = administrations.iloc[i][self.time_name]
                    time_next = administrations.iloc[i+1][self.time_name]
                    admin_measures = measures[(measures[self.time_name]>time_previous) & (measures[self.time_name]<=time_next)]
                    admin_col = np.full((len(admin_measures),1), adm_count)
                    admin_col = np.hstack((admin_col, admin_measures[[self.time_name,self.conc_name]].values)) # normalize the time and concentration
                    self.conc[p] = np.vstack((self.conc[p], admin_col))
                    adm_count += 1

                # get the measures after the last administration
                admin_measures = measures[measures[self.time_name]>time_next]
                admin_col = np.full((len(admin_measures),1), adm_count)
                admin_col = np.hstack((admin_col, admin_measures[[self.time_name,self.conc_name]].values)) # normalize the time and concentration

                self.conc[p] = np.vstack((self.conc[p], admin_col))
                self.conc[p] = self.conc[p][1:]
            else: # if there is only one administration
                admin_col = np.full((len(measures),1), adm_count)
                admin_col = np.hstack((admin_col, measures[[self.time_name,self.conc_name]].values))
                self.conc[p] = admin_col
            
            # get the administrations and normalize time and dose
            self.admin[p] = administrations[[self.dose_name,self.time_name]].values.astype(float)

            self.conc[p] = self.conc[p].astype(float) # convert to float

            self.time_infusion[p] = administrations[self.col_names[1]].values.astype(float) # get the infusion time

            # get the covariates as numpy float array
            if self.cov_names is not None:
                self.covariates[p] = (pat_data.iloc[0][self.cov_names].values).astype(float)
                self.covariates[p][-1] = self.covariates[p][-1]
        pass


    def convert_to_tensor(self):
        '''
        Convert the data to tensor format
        '''
        if isinstance(self.conc[self.patients[0]],torch.Tensor):
            print('Data is already in tensor format')
            return
        for p in self.patients:
            self.conc[p] = torch.tensor(self.conc[p], dtype=torch.float32)
            self.admin[p] = torch.tensor(self.admin[p], dtype=torch.float32)
            if self.cov_names is not None:
                self.covariates[p] = torch.tensor(self.covariates[p], dtype=torch.float32)
            self.time_infusion[p] = torch.tensor(self.time_infusion[p], dtype=torch.float32)
        pass


        def convert_to_array(self):
            '''
            Convert the data to numpy array format
            '''
            if isinstance(self.conc[self.patients[0]],np.ndarray):
                print('Data is already in numpy array format')
                return
            for p in self.patients:
                self.conc[p] = self.conc[p].numpy().astype(float)
                self.admin[p] = self.admin[p].numpy().astype(float)
                self.covariates[p] = self.covariates[p].numpy().astype(float)
                self.time_infusion[p] = self.time_infusion[p].numpy().astype(float)
            pass



class Synthetic_Data():
    '''
    Class to generate synthetic data
    '''

    def __init__(self, params, doses):
        self.params = params
        self.doses = doses
        pass

    def generate_data(self, n_points = 50, t_dur = 3):
        '''
        Generate the synthetic data for the given parameters and doses
        '''
        data_admin = []
        time = np.logspace(0, t_dur, num=n_points) - 1 # time points spaced logarithmically where to sample the data

        for d in self.doses:
            V = self.params[3]
            data_admin.append(odeint(utils.two_comp, [d/V,0.], time, args=(*self.params[:-1],))[:,0])
        
        # add a zero dose to to improve generalization if not already present
        if 0. not in self.doses:
            data_admin.append(np.zeros_like(data_admin[0])) 
            self.doses.append(0.)
        
        data_admin = np.array(data_admin)
        self.data_admin = data_admin
        self.time = time
        pass

    def convert_to_tensor(self):
        '''
        Convert the data to tensor format
        '''
        if isinstance(self.data_admin[0],torch.Tensor):
            print('Data is already in tensor format')
            return
        self.data_admin = torch.from_numpy(self.data_admin).float()
        self.time = torch.tensor(self.time, dtype=torch.float32)
        pass


class NODE(nn.Module):
    
    def __init__(self, dim_c, dim_V=None, n_cov=0):
        super().__init__()
        
        # Define the neural network
        self.n_cov = n_cov
        self.dim_c = dim_c
        self.dim_V = dim_V
        # Check if covariates are included
        if dim_V is not None:
            self.include_cov = True
        else:
            self.include_cov = False
        layers_c = []
        layers_V = []
        in_features_c = 4 + n_cov if self.include_cov else 4
        in_features_V = n_cov
        output_size = 1

        # Add hidden layers
        for d in dim_c:
            layers_c.append(nn.Linear(in_features_c, d))
            layers_c.append(nn.Softplus())
            in_features_c = d
        
        if self.include_cov:
            for d in dim_V:
                layers_V.append(nn.Linear(in_features_V, d))
                layers_V.append(nn.Softplus())
                in_features_V = d
        else:
            self.V = nn.Parameter(torch.tensor([1.], dtype=torch.float32, requires_grad=True))

        # Output layer
        layers_c.append(nn.Linear(in_features_c, output_size))
        if self.include_cov:
            layers_V.append(nn.Linear(in_features_V, output_size))
        
        # Combine all layers into a Sequential module
        self.net_c = nn.Sequential(*layers_c)
        if self.include_cov:
            self.net_V = nn.Sequential(*layers_V) 


    def forward(self,t,state):
        '''
        Forward pass of the model concatenating the state, time, initial condition, administrations and covariates
        '''
        t = torch.tensor([t])
        if self.include_cov:
            s = torch.cat([state,t,self.init_cond,self.admin,self.cov])
        else:
            s = torch.cat([state,t,self.init_cond,self.admin])
        return self.net_c(s)


    def get_cov(self, cov):
        # get the covariates
        self.cov = cov
        pass
    
    def init_dose(self, init):
        # get the initial dose
        self.init_cond = torch.tensor([init],dtype=torch.float32)
        pass
    
    def get_admin(self,n_admin):
        # get the number of previous administrations
        self.admin = (torch.tensor([n_admin], dtype=torch.float32))/10
    
    def train_synthetic(self, synth_data, covariates, epochs, lr, weights_decay):
        '''
        Train the model with synthetic data
        '''
        # get the list of patients to randomly select covaraites for each synthetic curve, curves are indipendent from covariates
        patients = list(covariates.keys())

        # Define the optimizer and loss function
        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weights_decay)
        MSE = nn.MSELoss()
        lossi_1adm = []

        start = time.time()
        for epoch in range(epochs):

            # loop over the doses present in the data
            for i in range(len(synth_data.doses)):
                
                optim.zero_grad()

                # setting the inital conditions
                self.init_dose(synth_data.data_admin[i][0])
                self.get_admin(1.)
                if self.include_cov:
                    random_patient = np.random.choice(patients)
                    self.get_cov(covariates[random_patient])
                init = self.init_cond*self.net_V(self.cov) if self.include_cov else self.init_cond/self.V

                # integrate the ODE
                pred_state = torchdiffeq.odeint(self, init, synth_data.time, method = 'dopri5')
                
                # calculate the loss and backpropagate
                loss = MSE(pred_state, synth_data.data_admin[i].view(-1,1))
                loss.backward()
                optim.step()
                lossi_1adm.append(loss.item())
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():7f}. Estimated remaning time: {(epochs/(epoch+1)-1)*(time.time()-start)/60:.3f} min')

        return lossi_1adm
    

    def train(self, data, epochs, lr, weights_decay):
        '''
        Train the model with real data
        '''
        # Define the optimizer and loss function
        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weights_decay)
        MSE = nn.MSELoss()
        lossi = []
        
        start = time.time()
        for epoch in range(epochs):

            for p in tqdm(data.patients):
        
                optim.zero_grad()
                
                # integrate the ODE
                pred_state = utils.compose_inegration_1d_for_data(data, p, self, include_covariates=self.include_cov)

                # calculate the loss and backpropagate
                loss = MSE(pred_state,data.conc[p][:,2].view(-1,1))
                loss.backward()
                lossi.append(loss.item())
                optim.step()

            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():7f}. Estimated remaning time: {(epochs/(epoch+1)-1)*(time.time()-start)/60:.3f} min')
        
        return lossi
    

    def plot_patient(self, data, pat):
        '''
        Plot the prediction for a specific patient
        '''
        with torch.no_grad():
            time, sol = utils.compose_inegration_1d_to_plot(data, pat, self, include_covariates=self.include_cov)
        plt.title(f'Prediction for {pat}')
        plt.xlabel('time')
        plt.ylabel('concentration')
        plt.grid(alpha=0.3)
        plt.plot(time, sol, label='Prediction')
        plt.show()
        pass

    def save(self, path):
        '''
        Save the model
        '''
        path_c = os.path.join(path, 'net_c.pth')
        torch.save(self.net_c.state_dict(), path_c)
        path_V = os.path.join(path, 'net_V.pth')
        if self.include_cov:
            torch.save(self.net_V.state_dict(), path_V)
        dims = {'dim_c': self.dim_c, 'dim_V': self.dim_V, 'n_cov': self.n_cov}
        with open(os.path.join(path, 'dims.pkl'), 'wb') as f:
            pickle.dump(dims, f)
        pass
    

    def load(self, path):
        '''
        Load the model
        '''
        path_c = os.path.join(path, 'net_c.pth')
        self.net_c.load_state_dict(torch.load(path_c,weights_only=True))
        path_V = os.path.join(path, 'net_V.pth')
        if self.include_cov:
            self.net_V.load_state_dict(torch.load(path_V,weights_only=True))
        pass
