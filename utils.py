
import pandas as pd
import numpy as np

import models
import __main__

from scipy.integrate import odeint as odeint_scipy
from scipy.optimize import curve_fit

import torch
from torchdiffeq import odeint

import pickle
import os


def check_col_names(col_names, default_col_names):
    '''
    Check if the column names are correct
    '''
    for name in default_col_names:
        if name in default_col_names:
            continue
        else:
            raise ValueError(f'The column {name} is not present in the dataset!')
    return True


def get_covariates_col(col_names, default_col_names):
    '''
    Get the covariates columns
    '''
    covariates = []
    for name in col_names:
        if name not in default_col_names:
            covariates.append(name)
    # remove columns with prefix 'Unnamed'
    prefix = 'Unnamed'
    remove_cols = [col for col in covariates if col.startswith(prefix)]
    for col in remove_cols:
        covariates.remove(col)
    return covariates


def create_data_class(data_csv, col_names, covariates):
    '''
    Create the data class
    '''
    data = models.Data(data_csv, col_names, covariates)
    data.create_data()
    return data


def get_doses(data):
    '''
    Get the unique doses from the data
    '''
    doses = []
    for p in data.patients:
        doses = doses + (data.admin[p][:,0]).tolist()
    return list(set(doses))


def two_comp(state,time,k0,k12,k21):
    ''''
    Two compartment model
    '''
    d_c1 = -k0*state[0] - k12*state[0] + k21*state[1]
    d_c2 = k12*state[0] - k21*state[1]
    return np.array([d_c1,d_c2])


def solve_ode_compose(p, k0, k12, k21, V):
    '''
    Solve the ODE for each patient separately, the solution is concatenated (not in this function)
    '''
    data = __main__.data # get the data from the main script
    
    conc = data.conc[p] # get the concentration data for the patient
    last_admin = int(conc[-1,0]) # get the number of administrations
    
    times = data.admin[p][:,1] # get the times of administrations
    doses = data.admin[p][:,0] # get the doses of administrations
    
    sol_tot = np.array([])
    
    # loop over the administrations
    for i in range(last_admin):
        
        t = conc[conc[:,0]==i+1][:,1]
        y0 = np.array([doses[i]/V,0.]) # initial conditions
        
        if i == 0 and last_admin > 1:
            t = np.concatenate([times[i:i+1], t, times[i+1:i+2]])
            # solve the ODE for the first administration
            sol = odeint_scipy(two_comp, y0, t, args = (k0,k12,k21))
        elif i<last_admin-1:
            t = np.concatenate([times[i:i+1], t, times[i+1:i+2]])
            y0 += sol[-1]
            # solve the ODE for the administrations in between
            sol = odeint_scipy(two_comp, y0, t, args = (k0,k12,k21))
        else:
            t = np.concatenate([times[i:i+1], t, t[-1].reshape(-1)+1.])
            y0 = y0 + sol[-1] if last_admin > 1 else y0
            # solve the ODE for the last administration
            sol = odeint_scipy(two_comp, y0, t, args = (k0,k12,k21))
            
        # concatenate the solutions for each administration
        if len(sol) > 2:
            sol_tot = np.concatenate([sol_tot, sol[1:-1,0]])
        else:
            continue

    return sol_tot


def fit_ode(t, k0, k12, k21, V):
    '''
    Wrapping function to fit the ODE parameters with scipy.curve_fit.
    It concatenates the solutions for each patient and returns the concatenated solution.
    '''
    
    sol_compose = np.array([])
    
    for p in __main__.data.patients:
        
        sol = solve_ode_compose(p, k0,k12,k21,V)
        sol_compose = np.concatenate([sol_compose,sol])
        
    return sol_compose


def get_all_times_and_conc(data):
    '''
    Get all the times and concentrations for all patients concatenated
    '''
    all_times = []
    all_conc = []
    for p in data.patients:
        d_pat = data.conc[p]
        all_times = all_times + (d_pat[:,1]).tolist()
        all_conc = all_conc + (d_pat[:,2]).tolist()
        
    all_times = np.array(all_times).astype(float)
    all_conc = np.array(all_conc).astype(float)
    
    return all_times, all_conc


def get_ode_params(data):
    '''
    Perform the ODE fitting
    '''
    all_times, all_conc = get_all_times_and_conc(data)
    
    p0 = np.array([1.,10.,10.,10.]) # initial guess for the parameters
    popt, pcov = curve_fit(fit_ode, all_times, all_conc, p0 = p0)
    
    return popt


def generate_sythetic_data(params, doses):
    '''
    Generate synthetic data
    '''
    synth_data = models.Synthetic_Data(params, doses)
    synth_data.generate_data()

    return synth_data


def compose_inegration_1d_for_data(data, pat, node, include_covariates):
    '''
    Compose the integration of the NODE for each patient separately
    '''
    # get the data from the spcified patient
    conc = data.conc[pat]
    admins = data.admin[pat]
    cov = data.covariates[pat]
    last_admin = int(conc[-1,0])
    if include_covariates:
        node.get_cov(cov)
    sol_tot = torch.tensor([])
    
    # loop over the administrations to the concatanate the solutions
    for j in range(last_admin):
        
        # indices of the concentrations for the current administration
        ind_j = (conc[:,0]==j+1).nonzero(as_tuple=True)[0]
        
        # get the times of the concentrations for the current administration
        if len(ind_j)>0:    
            tim = conc[ind_j,1]
        else:
            tim = torch.tensor([])
        
        # adding the times of the previous and next administrations
        if j<last_admin-1:
            tim = torch.cat([admins[j,1:2],tim,(admins[j+1,1:2]+0.01)])
        else:
            tim = torch.cat([admins[j,1:2],tim])
        
        # get the dose of the current administration and setting the initial values
        admin = admins[j,0].item()
        node.init_dose(admin)
        node.get_admin(j+1)
        conc_init = node.init_cond*node.net_V(node.cov) if include_covariates else node.init_cond/node.V
        if j>0:
            conc_init += solution[-1]

        # solve the ODE for the current administration
        solution = odeint(node, conc_init, tim-tim[0], method = 'dopri5') #dopri5
        
        # concatenate the solutions for each administration and remove the added points
        if j < last_admin-1:
            sol_tot = torch.cat([sol_tot,solution[1:-1]])
        else:
            sol_tot = torch.cat([sol_tot,solution[1:]])
        
    return sol_tot


def compose_inegration_1d_to_plot(data, pat, node, include_covariates):
    '''
    Compose the integration of the NODE for each patient separately for plotting
    '''
    admins = data.admin[pat]
    cov = data.covariates[pat]
    last_admin = len(admins)
    if include_covariates:
        node.get_cov(cov)
    sol_tot = torch.tensor([])
    time_tot = torch.tensor([])

    for j in range(last_admin-1):
        
        # indices of the concentrations for the current administration
        
        # get the times of the concentrations for the current administration
        t_range = [admins[j,1].item(),admins[j+1,1].item()]
        tim = torch.linspace(t_range[0],t_range[1],200)
        
        # get the dose of the current administration and setting the initial values
        admin = admins[j,0].item()
        node.init_dose(admin)
        node.get_admin(j+1)
        conc_init = node.init_cond*node.net_V(node.cov) if include_covariates else node.init_cond/node.V
        if j>0:
            conc_init += solution[-1]

        # solve the ODE for the current administration
        solution = odeint(node, conc_init, tim-tim[0], method = 'dopri5') #dopri5
        
        # concatenate the solutions for each administration and remove the added points
        sol_tot = torch.cat([sol_tot,solution[:-1]])
        time_tot = torch.cat([time_tot,tim[:-1]])

    # same as above but for the last administration
    tim = torch.linspace(admins[-1,1].item(),admins[-1,1].item()+1.5,200)
    admin = admins[-1,0].item()
    node.init_dose(admin)
    node.get_admin(last_admin)
    conc_init = node.init_cond*node.net_V(node.cov) if include_covariates else node.init_cond/node.V
    if last_admin>1:
        conc_init += solution[-1]
    solution = odeint(node, conc_init, tim-tim[0], method = 'dopri5')
    sol_tot = torch.cat([sol_tot,solution])
    time_tot = torch.cat([time_tot,tim])
    
    return time_tot.numpy(), sol_tot.numpy()


def load_NODE(path):
    '''
    Load the NODE model
    '''
    path_dims = os.path.join(path,'dims.pkl')
    with open(path_dims, 'rb') as f:
        dims = pickle.load(f)
    node = models.NODE(dims['dim_c'], dims['dim_V'], dims['n_cov'])
    node.load(path)
    print('NODE model loaded with parametes!\n')
    return node