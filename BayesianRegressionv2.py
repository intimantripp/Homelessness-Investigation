#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:41:43 2019

@author: intimantripp
"""

import pystan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity


## RUN EXPLORATORY ANALYSIS SCRIPT FIRST ##
# Many variables from that script are utilised in this one without being defined#


##################################################
           # 1. Data Preparations #
##################################################  
        
path = '/Users/intimantripp/Desktop/Documents/University of Edinburgh/Summer Projects/Project 2/Data/'        
###### 1. Load data ####### 
           
neet_data = pd.read_csv(path + 'neet.csv') # Not using
simd_data = pd.read_csv(path + 'simd_data.csv')
house_options_data = pd.read_csv(path + 'housing_options_data.csv')
drug_deaths_data = pd.read_csv(path + 'drug_related_deaths.csv')
earnings_data = pd.read_csv(path + 'earnings.csv')
pupil_attainment_data = pd.read_csv(path + 'pupil_attainment.csv')
fuel_poverty_data = pd.read_csv(path + 'fuel_poverty_data.csv')
drug_discharge_data = pd.read_csv(path + 'drug_related_discharge_data.csv')


simd_data.iloc[:,2:].boxplot()
plt.xticks(rotation=90)
plt.ylabel('Council SIMD rank')
plt.title('Boxplots of council SIMD rankings from 2003 to 2018')



###### 2. Restructure/Impute data ########

drug_discharge_data = drug_discharge_data[drug_discharge_data['Year']>2002]
house_options_data = house_options_data[house_options_data['Year']>2002]

neet_data.plot(x = 'Year', y = neet_data.columns[1:-1])
drug_deaths_data.plot.scatter(x = 'Year', y = 'Scotland')
pupil_attainment_data.plot.scatter(x = 'Year', y = 'Scotland')
fuel_poverty_data.plot.scatter(x = 'Year', y = 'Scotland')
drug_discharge_data.plot.scatter(x = 'Year', y = 'Scotland')
           


# Incomplete Data Analysis

def LinearRegressionImp(X_train, y, X_test):
    y = Convert2Array(y)
    reg = LinearRegression()
    reg.fit(X_train, y)
    predictions = reg.predict(X_test)
    return predictions

def Convert2Array(series):
    n = series.size
    array = np.array(series).reshape(n, 1)
    return array


def ImputeObservations(df, column_headings, X_train, y_train_ind, X_test, year, data_type):
    # Impute future observations
    results = [year]
    for heading in column_headings:
        print(heading)
        prediction = LinearRegressionImp(X_train, df[heading][y_train_ind:], X_test)
        results.append(prediction)
    series = pd.Series(results).astype(data_type)
    return np.array(series)

#1. drug death data - Impute values for year 2018 using linear regression

drug_deaths_data.loc[drug_deaths_data.shape[0]] = ImputeObservations(drug_deaths_data, 
                    drug_deaths_data.columns[1:], 
                    np.array(drug_deaths_data['Year'][-5:]).reshape(5,1), -5, 
                    np.array(2018).reshape(1,1), 2018, int)


for council in drug_deaths_data.columns:
    plt.scatter(drug_deaths_data['Year'][:-1], drug_deaths_data[council][:-1], color = 'b',
                label = 'Original')
    plt.scatter(drug_deaths_data['Year'].iloc[-1], drug_deaths_data[council].iloc[-1], 
                color = 'r', label = 'Imputed')
    plt.legend()
    plt.title(council)
    plt.ylabel('Number of deaths')
    plt.xlabel('Year')
    plt.show()

drug_deaths_data[drug_deaths_data<0] = 0
drug_deaths_data = drug_deaths_data[drug_deaths_data['Year']>2002]




#2. Pupil attainment data - Impute values for 2013 onwards using linear regression

# appears extremely quadratic
for council in pupil_attainment_data.columns:
    plt.scatter(pupil_attainment_data['Year'], pupil_attainment_data[council])
    plt.show()

for i in range(5):
    pupil_attainment_data.loc[pupil_attainment_data.shape[0]] = ImputeObservations(pupil_attainment_data,
                             pupil_attainment_data.columns[1:],
                   np.array(pupil_attainment_data['Year'][-10:]).reshape(10,1), -10, np.array(2014+i).reshape(1,1), 2014+i, float)

for council in pupil_attainment_data.columns:
    plt.scatter(pupil_attainment_data[pupil_attainment_data['Year']<2014]['Year'], 
                pupil_attainment_data[pupil_attainment_data['Year']<2014][council], color='b',
                label = 'Original')
    plt.scatter(pupil_attainment_data[pupil_attainment_data['Year']>=2014]['Year'], 
                pupil_attainment_data[pupil_attainment_data['Year']>=2014][council], 
                color='r', label = 'Imputed')
    plt.title(council)
    plt.ylabel('Percentage')
    plt.xlabel('Year')
    plt.legend()
    plt.show()


#3. Fuel Poverty - impute values for 2010 before using linear regression

def ImputeObservations2(df, column_headings, X_train, y_train_ind, X_test, year, data_type):
    
    # Impute past observations
    results = [year]
    for heading in column_headings:
        prediction = LinearRegressionImp(X_train, df[heading][:y_train_ind], X_test)
        results.append(prediction)
    series = pd.Series(results).astype(data_type)
    return np.array(series)


for council in fuel_poverty_data.columns:
    plt.scatter(fuel_poverty_data['Year'], fuel_poverty_data[council])
    plt.show()

fuel_poverty_data = pd.read_csv(path + 'fuel_poverty_data.csv')

fuel_poverty_data.loc[fuel_poverty_data.shape[0]] = ImputeObservations(fuel_poverty_data, 
                     fuel_poverty_data.columns[1:], 
                     np.array(fuel_poverty_data['Year'][-5:]).reshape(  5,1), -5, 
                     np.array(2018).reshape(1,1), 2018, float)

for i in range(7):
    X_train = np.array(fuel_poverty_data['Year'][-9-i:-1-i]).reshape((8, 1))
    fuel_poverty_data.loc[-1] = ImputeObservations2(fuel_poverty_data, fuel_poverty_data.columns[1:], 
                             X_train, 8, np.array(2009-i).reshape(1,1),
                             2009-i, float)
    fuel_poverty_data.index = fuel_poverty_data.index + 1
    fuel_poverty_data.sort_index(inplace = True)

for council in fuel_poverty_data.columns:
    plt.scatter(fuel_poverty_data['Year'][7:-1], fuel_poverty_data[council][7:-1], 
                color = 'b', label = 'Original')
    plt.scatter(fuel_poverty_data['Year'][:7], fuel_poverty_data[council][:7], color = 'r',
                label = 'Imputed')
    plt.scatter(fuel_poverty_data['Year'][-1:], fuel_poverty_data[council][-1:], color = 'r')
    plt.legend()
    plt.ylabel('Percentage')
    plt.xlabel('Year')
    plt.title(council)
    plt.show()



## Define a regression dictionary with years 2003-2018
    
council_reg_dict = {council:council_dict_yrs[council][council_dict_yrs[council]['Year']>2002] for council in list(council_dict_yrs.keys())}
council_reg_dict = {council:council_reg_dict[council][council_reg_dict[council]['Year']<2019] for council in list(council_reg_dict.keys())}


## Make year columns integers again

for data_frame in [simd_data, house_options_data, earnings_data, drug_deaths_data, pupil_attainment_data, fuel_poverty_data, drug_discharge_data]:
    data_frame['Year'] = data_frame['Year'].astype(int)



######  2. Combine datasets #######

def AddCouncilData(council_dict, data_list):
    
    council_list = list(council_dict.keys())
    column_heading_list = ['SIMD', 'housing_options', 'earnings', 'drug_deaths', 'pupil_attainment', 
                           'fuel_poverty', 'drug_discharge']
    for i in range(len(council_list)):
        for j in range(len(data_list)):
            council_dict[council_list[i]] = council_dict[council_list[i]].merge(data_list[j][['Year', 
                        council_list[i]]], how = 'left')
            council_dict[council_list[i]] = council_dict[council_list[i]].rename(columns = {council_dict[council_list[i]].columns[-1]: column_heading_list[j]})
        
    return council_dict
    
        
def CleanCouncilDict(council_dict):
    
    for council in list(council_dict.keys()):
        if council == 'Scotland':
            council_dict[council] = council_dict[council].drop(columns = ['Month'])
            break

        council_dict[council] = council_dict[council].drop(columns = ['Council', 'Month'])
        
    return council_dict

    

council_reg_dict = AddCouncilData(council_reg_dict,
                                  [simd_data, house_options_data, earnings_data, drug_deaths_data,
                                   pupil_attainment_data, fuel_poverty_data, drug_discharge_data])
council_reg_dict = CleanCouncilDict(council_reg_dict)




##################################################
      # 2. Bayesian Poisson Modelling #
##################################################  
      

###### 1. Model Strings    ######

bayesian_baseline_model_code = """


    data {
        int n;
        int y[n];
    }
    
    parameters {
        real alpha;
    }
    
    transformed parameters {
        vector<lower=0>[n] lambda;
        
        for (i in 1:n) {
                lambda[i] = exp(alpha);
        }
        
    }
    
    model {
        alpha ~ normal(0, 10);
        
        for (i in 1:n){
            y[i] ~ poisson(lambda[i]);
        }
    }

"""      

bayesian_full_model_code = """

    data {
        int n;
        int y[n];
        vector[n] SIMD;
        vector[n] drug_deaths;
        vector[n] earnings;
        vector[n] pupil_attainment;
        vector[n] fuel_poverty;
        vector[n] pop;
        vector[n] drug_discharge;
        vector[n] housing_options;
        real SIMD_bar;
        real drug_deaths_bar;
        real earnings_bar;
        real pupil_attainment_bar;
        real fuel_poverty_bar;
        real pop_bar;
        real drug_discharge_bar;
        real housing_options_bar;
        real SIMD_sd;
        real drug_deaths_sd;
        real earnings_sd;
        real pupil_attainment_sd;
        real fuel_poverty_sd;
        real pop_sd;
        real drug_discharge_sd;
        real housing_options_sd;
    }
    
    parameters {
        real alpha;
        real beta0;
        real beta1;
        real beta2;
        real beta3;
        real beta4;
        real beta5;
        real beta6;
        real beta7;      
    }
    
    transformed parameters {
        vector<lower=0>[n] lambda;
        
        for (i in 1:n) {
                lambda[i] = exp(alpha + beta0*(SIMD[i] - SIMD_bar)/SIMD_sd + 
                beta1*(drug_deaths[i] - drug_deaths_bar) + beta2*(earnings[i] - earnings_bar)/earnings_sd + 
                beta3*(pupil_attainment[i]-pupil_attainment_bar)/pupil_attainment_sd + 
                beta4*(fuel_poverty[i] - fuel_poverty_bar)/fuel_poverty_sd
                + beta5*(pop[i] - pop_bar)/pop_sd + 
                beta6*(drug_discharge[i] - drug_discharge_bar)/drug_discharge_sd + 
                beta7*(housing_options[i] - housing_options_bar)/housing_options_sd);
        }
        
    }
    
    model {
        alpha ~ normal(0, 10);
        beta0 ~ normal(0, 10);
        beta1 ~ normal(0, 10);
        beta2 ~ normal(0, 10);
        beta3 ~ normal(0, 10);
        beta4 ~ normal(0, 10);
        beta5 ~ normal(0, 10);
        beta6 ~ normal(0, 10);
        beta7 ~ normal(0, 10);
        
        for (i in 1:n){
            y[i] ~ poisson(lambda[i]);
        }
    }
"""

bayesian_reduced_model_code = """


    data {
        int n;
        int y[n];
        vector[n] SIMD;
        vector[n] earnings;
        vector[n] pupil_attainment;
        vector[n] fuel_poverty;
        vector[n] pop;
        vector[n] drug_discharge;
        real SIMD_bar;
        real pupil_attainment_bar;
        real earnings_bar;
        real earnings_sd;        
        real fuel_poverty_bar;
        real pop_bar;
        real drug_discharge_bar;
        real SIMD_sd;
        real pupil_attainment_sd;
        real fuel_poverty_sd;
        real pop_sd;
        real drug_discharge_sd;
    }
    
    parameters {
        real alpha;
        real beta0;
        real beta2;
        real beta3;
        real beta4;
        real beta5;
        real beta6;     
    }
    
    transformed parameters {
        vector<lower=0>[n] lambda;
        
        for (i in 1:n) {
                lambda[i] = exp(alpha + beta0*(SIMD[i] - SIMD_bar)/SIMD_sd + 
                beta2*(earnings[i] - earnings_bar)/earnings_sd + 
                beta3*(pupil_attainment[i]-pupil_attainment_bar)/pupil_attainment_sd + 
                beta4*(fuel_poverty[i] - fuel_poverty_bar)/fuel_poverty_sd
                + beta5*(pop[i] - pop_bar)/pop_sd + 
                beta6*(drug_discharge[i] - drug_discharge_bar)/drug_discharge_sd);
        }
        
    }
    
    model {
        alpha ~ normal(0, 10);
        beta0 ~ normal(0, 10);
        beta2 ~ normal(0, 10);
        beta3 ~ normal(0, 10);
        beta4 ~ normal(0, 10);
        beta5 ~ normal(0, 10);
        beta6 ~ normal(0, 10);
        
        for (i in 1:n){
            y[i] ~ poisson(lambda[i]);
        }
    }
"""

bayesian_reduced_model_code = """


    data {
        int n;
        int y[n];
        vector[n] SIMD;
        vector[n] earnings;
        vector[n] pupil_attainment;
        vector[n] fuel_poverty;
        vector[n] pop;
        vector[n] drug_discharge;
        real SIMD_bar;
        real pupil_attainment_bar;
        real earnings_bar;
        real earnings_sd;        
        real fuel_poverty_bar;
        real pop_bar;
        real drug_discharge_bar;
        real SIMD_sd;
        real pupil_attainment_sd;
        real fuel_poverty_sd;
        real pop_sd;
        real drug_discharge_sd;
    }
    
    parameters {
        real alpha;
        real beta3;
        real beta4;
        real beta5;
        real beta6;     
    }
    
    transformed parameters {
        vector<lower=0>[n] lambda;
        
        for (i in 1:n) {
                lambda[i] = exp(alpha + 
                beta3*(pupil_attainment[i]-pupil_attainment_bar)/pupil_attainment_sd + 
                beta4*(fuel_poverty[i] - fuel_poverty_bar)/fuel_poverty_sd
                + beta5*(pop[i] - pop_bar)/pop_sd + 
                beta6*(drug_discharge[i] - drug_discharge_bar)/drug_discharge_sd);
        }
        
    }
    
    model {
        alpha ~ normal(0, 10);
        beta3 ~ normal(0, 10);
        beta4 ~ normal(0, 10);
        beta5 ~ normal(0, 10);
        beta6 ~ normal(0, 10);
        
        for (i in 1:n){
            y[i] ~ poisson(lambda[i]);
        }
    }
"""

      
###### 2. Model Validations   ######

years = list(range(2003, 2019))
os.chdir('/Users/intimantripp/Desktop/Documents/University of Edinburgh/Summer Projects/Project 2/figures/validations')

def CalcPredictionsVal (bayes_res_dict, X, X_train):
    results = np.empty((10000, len(X)))
    
    SIMD_bar = np.mean(X['SIMD']); SIMD_sd = np.std(X_train['SIMD'])
    drug_deaths_bar = np.mean(X['drug_deaths']); drug_deaths_sd = np.std(X_train['drug_deaths'])
    earnings_bar = np.mean(X['earnings']); earnings_sd = np.std(X_train['earnings'])
    pupil_att_bar = np.mean(X['pupil_attainment']); pupil_att_sd = np.std(X_train['pupil_attainment'])
    fuel_pov_bar = np.mean(X['fuel_poverty']); fuel_pov_sd = np.std(X_train['fuel_poverty'])
    pop_bar = np.mean(X['pop']); pop_sd = np.std(X_train['pop'])
    drug_disch_bar = np.mean(X['drug_discharge']); drug_disch_sd = np.std(X_train['drug_discharge'])
    house_opt_bar = np.mean(X['housing_options']); house_opt_sd = np.std(X_train['housing_options'])

    for i in range(len(X)):
        SIMD = X['SIMD'].iloc[i]
        drug_deaths = X['drug_deaths'].iloc[i]
        earnings = X['earnings'].iloc[i]
        pupil_att = X['pupil_attainment'].iloc[i]
        fuel_pov = X['fuel_poverty'].iloc[i]
        pop = X['pop'].iloc[i]
        drug_disch = X['drug_discharge'].iloc[i]
        house_opt = X['housing_options'].iloc[i]
        results[:, i] = np.exp(bayes_res_dict['alpha'] + 
                        bayes_res_dict['beta0']*(SIMD - SIMD_bar)/SIMD_sd + 
                        bayes_res_dict['beta1']*(drug_deaths - drug_deaths_bar)/drug_deaths_sd + 
                        bayes_res_dict['beta2']*(earnings - earnings_bar)/earnings_sd + 
                        bayes_res_dict['beta3']*(pupil_att - pupil_att_bar)/pupil_att_sd + 
                        bayes_res_dict['beta4']*(fuel_pov - fuel_pov_bar)/fuel_pov_sd + 
                        bayes_res_dict['beta5']*(pop - pop_bar)/pop_sd + 
                        bayes_res_dict['beta6']*(drug_disch - drug_disch_bar)/drug_disch_sd + 
                        bayes_res_dict['beta7']*(house_opt - house_opt_bar)/house_opt_sd)
    
    return results

def CalcPredictionsReducedVal (bayes_res_dict, X, X_train):
    results = np.empty((10000, len(X)))
    
    SIMD_bar = np.mean(X['SIMD']); SIMD_sd = np.std(X_train['SIMD'])
    pupil_att_bar = np.mean(X['pupil_attainment']); pupil_att_sd = np.std(X_train['pupil_attainment'])
    fuel_pov_bar = np.mean(X['fuel_poverty']); fuel_pov_sd = np.std(X_train['fuel_poverty'])
    pop_bar = np.mean(X['pop']); pop_sd = np.std(X_train['pop'])
    drug_disch_bar = np.mean(X['drug_discharge']); drug_disch_sd = np.std(X_train['drug_discharge'])

    for i in range(len(X)):
        SIMD = X['SIMD'].iloc[i]
        pupil_att = X['pupil_attainment'].iloc[i]
        fuel_pov = X['fuel_poverty'].iloc[i]
        pop = X['pop'].iloc[i]
        drug_disch = X['drug_discharge'].iloc[i]
        results[:, i] = np.exp(bayes_res_dict['alpha'] + 
                        #bayes_res_dict['beta0']*(SIMD - SIMD_bar)/SIMD_sd + 
                        bayes_res_dict['beta3']*(pupil_att - pupil_att_bar)/pupil_att_sd + 
                        bayes_res_dict['beta4']*(fuel_pov - fuel_pov_bar)/fuel_pov_sd + 
                        bayes_res_dict['beta5']*(pop - pop_bar)/pop_sd + 
                        bayes_res_dict['beta6']*(drug_disch - drug_disch_bar)/drug_disch_sd)
    
    return results

def CalcPredictionsBaselineVal (bayes_res_dict, X):
    results = np.empty((10000, len(X)))

    for i in range(len(X)):
        results[:, i] = np.exp(bayes_res_dict['alpha'])
    
    return results

def PlotPredictionsVal(predictions_train, predictions_test, council, test_year):
    upper_bound_train = np.quantile(predictions_train, 0.9, axis = 0)
    lower_bound_train = np.quantile(predictions_train, 0.1, axis = 0)
    means_train = np.mean(predictions_train, axis = 0)
    
    upper_bound_test = np.quantile(predictions_test, 0.9, axis = 0)
    lower_bound_test = np.quantile(predictions_test, 0.1, axis = 0)
    means_test = np.mean(predictions_test, axis = 0)
    
    #True points
    plt.scatter(council_reg_dict[council]['Year'], 
                                 council_reg_dict[council]['Applications'], 
                label = 'Actual', color = 'b')
    #Predictions - Trained points
    plt.errorbar(council_reg_dict[council][council_reg_dict[council]['Year']!= test_year]['Year'], means_train, 
                 yerr = (means_train - lower_bound_train, upper_bound_train - means_train), label = 'Predicted - Train', fmt = 'x',
                 color = 'r')
    #Predictions - Test points
    plt.errorbar(council_reg_dict[council][council_reg_dict[council]['Year']==test_year]['Year'], means_test,
                 yerr = (means_test - lower_bound_test, upper_bound_test - means_test), label = 'Predicted - Test',
                 fmt = 'x', color = 'orange')
    
    
    plt.title(council)
    plt.xlabel('Year'); plt.ylabel('Number of applications')
    plt.legend()
    plt.savefig(council + ' - ' + str(test_year) + '_red.png')
    plt.show()  
    return

## 1. Full Model 

def run_bayesian_full_model_validation(): 
    council_rmse_dict = {}
    council_samples_dict = {}
    council_summary_dict = {}
    for council in council_reg_dict.keys():
        
        rmse_dict = {}
        samples_dict = {}
        summary_dict = {}
        for year in years[:1] + years[3:4] + years[10:11] + years[14:15] + years[15:]:
            X_train = council_reg_dict[council][council_reg_dict[council]['Year'] != year][['SIMD',
                                  'housing_options', 'earnings', 'drug_deaths', 'pupil_attainment', 
                                  'fuel_poverty', 'drug_discharge', 'pop']]
            X_test = council_reg_dict[council][council_reg_dict[council]['Year'] == year][['SIMD',
                                  'housing_options', 'earnings', 'drug_deaths', 'pupil_attainment', 
                                  'fuel_poverty', 'drug_discharge', 'pop']]
        
            y_train = council_reg_dict[council][council_reg_dict[council]['Year']!=year]['Applications']
            y_test = council_reg_dict[council][council_reg_dict[council]['Year']==year]['Applications']
            n = len(y_train)
            data = {
                    'n':n,
                    'pop':X_train['pop'], 'pop_bar':np.mean(X_train['pop']), 'pop_p':X_test['pop'].iloc[0], 'pop_sd':np.std(X_train['pop']),
                    'SIMD':X_train['SIMD'], 'SIMD_bar':np.mean(X_train['SIMD']), 'SIMD_p':X_test['SIMD'].iloc[0], 'SIMD_sd':np.std(X_train['SIMD']),
                    'drug_deaths':X_train['drug_deaths'], 'drug_deaths_bar':np.mean(X_train['drug_deaths']), 'drug_deaths_p':X_test['drug_deaths'].iloc[0], 'drug_deaths_sd':np.std(X_test['drug_deaths']),
                    'earnings':X_train['earnings'], 'earnings_bar':np.mean(X_train['earnings']), 'earnings_p':X_test['earnings'].iloc[0], 'earnings_sd':np.std(X_train['earnings']),
                    'pupil_attainment':X_train['pupil_attainment'], 'pupil_attainment_bar':np.mean(X_train['pupil_attainment']), 'pupil_attainment_p':X_test['pupil_attainment'].iloc[0], 'pupil_attainment_sd':np.std(X_train['pupil_attainment']),    
                    'fuel_poverty':X_train['fuel_poverty'], 'fuel_poverty_bar':np.mean(X_train['fuel_poverty']), 'fuel_poverty_p': X_test['fuel_poverty'].iloc[0], 'fuel_poverty_sd':np.std(X_train['fuel_poverty']),
                    'drug_discharge':X_train['drug_discharge'], 'drug_discharge_bar':np.mean(X_train['drug_discharge']), 'drug_discharge_p': X_test['drug_discharge'].iloc[0], 'drug_discharge_sd':np.std(X_train['drug_discharge']),
                    'housing_options':X_train['housing_options'], 'housing_options_bar':np.mean(X_train['housing_options']), 'housing_options_p':X_test['housing_options'].iloc[0], 'housing_options_sd':np.std(X_train['housing_options']),
                    'y': y_train,  
                    }

            fit = bayesian_full_model.sampling(data=data, iter=10000, chains=2, warmup=5000, init=0,
                                               seed = 42, control=dict(max_treedepth=13))
            summary = fit.summary()
            summary_df = pd.DataFrame(summary['summary'], columns = summary['summary_colnames'],
                                      index = summary['summary_rownames'])
            summary_dict[year] = summary_df
            
            samples = fit.extract()
            samples_dict[year] = samples
            predictions_train = CalcPredictionsVal(samples, X_train, X_train)
            predictions_test = CalcPredictionsVal(samples, X_test, X_train)
            rmse_dict[year] = np.sqrt(mean_squared_error(np.repeat(y_test,10000), predictions_test))
            PlotPredictionsVal(predictions_train, predictions_test, council, year)
        council_rmse_dict[council] = rmse_dict
        council_summary_dict[council] = summary_dict
        council_samples_dict[council] = samples_dict
    return  {'rmse_dicts':council_rmse_dict, 'summary_dicts':council_summary_dict,
             'samples_dicts':council_samples_dict}

bayesian_full_model = pystan.StanModel(model_code = bayesian_full_model_code)
bayesian_validation_model_full_results = run_bayesian_full_model_validation()


## 2. Reduced Model

def run_bayesian_reduced_model_validation(): 
    council_rmse_dict = {}
    council_samples_dict = {}
    council_summary_dict = {}
    for council in council_reg_dict.keys():
        
        rmse_dict = {}
        samples_dict = {}
        summary_dict = {}
        for year in years[:1] + years[3:4] + years[10:11] + years[14:15] + years[15:]:
            X_train = council_reg_dict[council][council_reg_dict[council]['Year'] != year][['SIMD',
                                  'housing_options', 'earnings', 'drug_deaths', 'pupil_attainment', 
                                  'fuel_poverty', 'drug_discharge', 'pop']]
            X_test = council_reg_dict[council][council_reg_dict[council]['Year'] == year][['SIMD',
                                  'housing_options', 'earnings', 'drug_deaths', 'pupil_attainment', 
                                  'fuel_poverty', 'drug_discharge', 'pop']]
        
            y_train = council_reg_dict[council][council_reg_dict[council]['Year']!=year]['Applications']
            y_test = council_reg_dict[council][council_reg_dict[council]['Year']==year]['Applications']
            n = len(y_train)
            data = {
                    'n':n,
                    'pop':X_train['pop'], 'pop_bar':np.mean(X_train['pop']), 'pop_p':X_test['pop'].iloc[0], 'pop_sd':np.std(X_train['pop']),
                    'SIMD':X_train['SIMD'], 'SIMD_bar':np.mean(X_train['SIMD']), 'SIMD_p':X_test['SIMD'].iloc[0], 'SIMD_sd':np.std(X_train['SIMD']),
                    'drug_deaths':X_train['drug_deaths'], 'drug_deaths_bar':np.mean(X_train['drug_deaths']), 'drug_deaths_p':X_test['drug_deaths'].iloc[0], 'drug_deaths_sd':np.std(X_test['drug_deaths']),
                    'earnings':X_train['earnings'], 'earnings_bar':np.mean(X_train['earnings']), 'earnings_p':X_test['earnings'].iloc[0], 'earnings_sd':np.std(X_train['earnings']),
                    'pupil_attainment':X_train['pupil_attainment'], 'pupil_attainment_bar':np.mean(X_train['pupil_attainment']), 'pupil_attainment_p':X_test['pupil_attainment'].iloc[0], 'pupil_attainment_sd':np.std(X_train['pupil_attainment']),    
                    'fuel_poverty':X_train['fuel_poverty'], 'fuel_poverty_bar':np.mean(X_train['fuel_poverty']), 'fuel_poverty_p': X_test['fuel_poverty'].iloc[0], 'fuel_poverty_sd':np.std(X_train['fuel_poverty']),
                    'drug_discharge':X_train['drug_discharge'], 'drug_discharge_bar':np.mean(X_train['drug_discharge']), 'drug_discharge_p': X_test['drug_discharge'].iloc[0], 'drug_discharge_sd':np.std(X_train['drug_discharge']),
                    'housing_options':X_train['housing_options'], 'housing_options_bar':np.mean(X_train['housing_options']), 'housing_options_p':X_test['housing_options'].iloc[0], 'housing_options_sd':np.std(X_train['housing_options']),
                    'y': y_train,  
                    }

            fit = bayesian_reduced_model.sampling(data=data, iter=10000, chains=2, warmup=5000, init=0,
                                               seed = 42, control=dict(max_treedepth=13))
            summary = fit.summary()
            summary_df = pd.DataFrame(summary['summary'], columns = summary['summary_colnames'],
                                      index = summary['summary_rownames'])
            summary_dict[year] = summary_df
            
            samples = fit.extract()
            samples_dict[year] = samples
            
            
            predictions_train = CalcPredictionsReducedVal(samples, X_train, X_train)
            predictions_test = CalcPredictionsReducedVal(samples, X_test, X_train)
            rmse_dict[year] = np.sqrt(mean_squared_error(np.repeat(y_test,10000), predictions_test))
            PlotPredictionsVal(predictions_train, predictions_test, council, year)
        council_rmse_dict[council] = rmse_dict
        council_samples_dict[council] = samples_dict
        council_summary_dict[council] = summary_dict
    return  {'rmse_dicts':council_rmse_dict, 'summary_dicts':council_summary_dict,
             'samples_dicts':council_samples_dict}

bayesian_reduced_model = pystan.StanModel(model_code = bayesian_reduced_model_code)
bayesian_validation_model_reduced_results = run_bayesian_reduced_model_validation()

## 3. Baseline model

def run_bayesian_baseline_model_validation(): 
    council_rmse_dict = {}
    council_samples_dict = {}
    council_summary_dict = {}
    for council in council_reg_dict.keys():
        
        rmse_dict = {}
        samples_dict = {}
        summary_dict = {}
        for year in years[:1] + years[3:4] + years[10:11] + years[14:15] + years[15:]:
            X_train = council_reg_dict[council][council_reg_dict[council]['Year'] != year][['SIMD',
                                  'housing_options', 'earnings', 'drug_deaths', 'pupil_attainment', 
                                  'fuel_poverty', 'drug_discharge', 'pop']]
            X_test = council_reg_dict[council][council_reg_dict[council]['Year'] == year][['SIMD',
                                  'housing_options', 'earnings', 'drug_deaths', 'pupil_attainment', 
                                  'fuel_poverty', 'drug_discharge', 'pop']]
        
            y_train = council_reg_dict[council][council_reg_dict[council]['Year']!=year]['Applications']
            y_test = council_reg_dict[council][council_reg_dict[council]['Year']==year]['Applications']
            n = len(y_train)
            data = {
                    'n':n,
                    'y': y_train,  
                    }

            fit = bayesian_baseline_model.sampling(data=data, iter=10000, chains=2, warmup=5000, init=0,
                                               seed = 42, control=dict(max_treedepth=13))
            summary = fit.summary()
            summary_df = pd.DataFrame(summary['summary'], columns = summary['summary_colnames'],
                                      index = summary['summary_rownames'])
            summary_dict[year] = summary_df
            
            samples = fit.extract()
            samples_dict[year] = samples
            
            
            predictions_train = CalcPredictionsBaselineVal(samples, X_train)
            predictions_test = CalcPredictionsBaselineVal(samples, X_test)
            rmse_dict[year] = np.sqrt(mean_squared_error(np.repeat(y_test,10000), predictions_test))
            PlotPredictionsVal(predictions_train, predictions_test, council, year)
            
        council_rmse_dict[council] = rmse_dict
        council_samples_dict[council] = samples_dict
        council_summary_dict[council] = summary_dict
    return  {'rmse_dicts':council_rmse_dict, 'summary_dicts':council_summary_dict,
             'samples_dicts':council_samples_dict}

bayesian_baseline_model = pystan.StanModel(model_code = bayesian_baseline_model_code)
bayesian_validation_model_baseline_results = run_bayesian_baseline_model_validation()

###### 2. Model Coefficients   ######

os.chdir('/Users/intimantripp/Desktop/Documents/University of Edinburgh/Summer Projects/Project 2/figures/predictions2')

def CalcPredictions (bayes_res_dict, X):
    results = np.empty((10000, len(X)))
    SIMD_bar = np.mean(X['SIMD']); SIMD_sd = np.std(X['SIMD'])
    drug_deaths_bar = np.mean(X['drug_deaths']); drug_deaths_sd = np.std(X['drug_deaths'])
    earnings_bar = np.mean(X['earnings']); earnings_sd = np.std(X['earnings'])
    pupil_att_bar = np.mean(X['pupil_attainment']); pupil_att_sd = np.std(X['pupil_attainment'])
    fuel_pov_bar = np.mean(X['fuel_poverty']); fuel_pov_sd = np.std(X['fuel_poverty'])
    pop_bar = np.mean(X['pop']); pop_sd = np.std(X['pop'])
    drug_disch_bar = np.mean(X['drug_discharge']); drug_disch_sd = np.std(X['drug_discharge'])
    house_opt_bar = np.mean(X['housing_options']); house_opt_sd = np.std(X['housing_options'])

    for i in range(len(X)):
        SIMD = X['SIMD'][i]
        drug_deaths = X['drug_deaths'][i]
        earnings = X['earnings'][i]
        pupil_att = X['pupil_attainment'][i]
        fuel_pov = X['fuel_poverty'][i]
        pop = X['pop'][i]
        drug_disch = X['drug_discharge'][i]
        house_opt = X['housing_options'][i]
        results[:, i] = np.exp(bayes_res_dict['alpha'] + 
                        bayes_res_dict['beta0']*(SIMD - SIMD_bar)/SIMD_sd + 
                        bayes_res_dict['beta1']*(drug_deaths - drug_deaths_bar)/drug_deaths_sd + 
                        bayes_res_dict['beta2']*(earnings - earnings_bar)/earnings_sd + 
                        bayes_res_dict['beta3']*(pupil_att - pupil_att_bar)/pupil_att_sd + 
                        bayes_res_dict['beta4']*(fuel_pov - fuel_pov_bar)/fuel_pov_sd + 
                        bayes_res_dict['beta5']*(pop - pop_bar)/pop_sd + 
                        bayes_res_dict['beta6']*(drug_disch - drug_disch_bar)/drug_disch_sd + 
                        bayes_res_dict['beta7']*(house_opt - house_opt_bar)/house_opt_sd)
    
    return results

def CalcPredictionsReduced (bayes_res_dict, X):
    results = np.empty((10000, len(X)))
    SIMD_bar = np.mean(X['SIMD']); SIMD_sd = np.std(X['SIMD'])
    earnings_bar = np.mean(X['earnings']); earnings_sd = np.std(X['earnings'])
    pupil_att_bar = np.mean(X['pupil_attainment']); pupil_att_sd = np.std(X['pupil_attainment'])
    fuel_pov_bar = np.mean(X['fuel_poverty']); fuel_pov_sd = np.std(X['fuel_poverty'])
    pop_bar = np.mean(X['pop']); pop_sd = np.std(X['pop'])
    drug_disch_bar = np.mean(X['drug_discharge']); drug_disch_sd = np.std(X['drug_discharge'])

    for i in range(len(X)):
        SIMD = X['SIMD'][i];
        drug_deaths = X['drug_deaths'][i]
        earnings = X['earnings'][i]
        pupil_att = X['pupil_attainment'][i]
        fuel_pov = X['fuel_poverty'][i]
        pop = X['pop'][i]
        drug_disch = X['drug_discharge'][i]
        house_opt = X['housing_options'][i]
        results[:, i] = np.exp(bayes_res_dict['alpha'] + 
                        #bayes_res_dict['beta0']*(SIMD - SIMD_bar)/SIMD_sd + 
                        #bayes_res_dict['beta2']*(earnings - earnings_bar)/earnings_sd + 
                        bayes_res_dict['beta3']*(pupil_att - pupil_att_bar)/pupil_att_sd + 
                        bayes_res_dict['beta4']*(fuel_pov - fuel_pov_bar)/fuel_pov_sd + 
                        bayes_res_dict['beta5']*(pop - pop_bar)/pop_sd + 
                        bayes_res_dict['beta6']*(drug_disch - drug_disch_bar)/drug_disch_sd)
    
    return results

def CalcPredictionsBaseline (bayes_res_dict, X):
    results = np.empty((10000, len(X)))

    for i in range(len(X)):
        results[:, i] = np.exp(bayes_res_dict['alpha'])
    
    return results

def PlotPredictions(results, X, council):
    upper_bound = np.quantile(results, 0.9, axis = 0)
    lower_bound = np.quantile(results, 0.1, axis = 0)
    means = np.mean(results, axis = 0)
    plt.scatter(council_reg_dict[council]['Year'], council_reg_dict[council]['Applications'], 
                label = 'Actual', color = 'b')
    plt.errorbar(council_reg_dict[council]['Year'], means, 
                 yerr = (means - lower_bound, upper_bound - means), label = 'Predicted', fmt = 'x',
                 color = 'r')
    plt.title(council)
    plt.xlabel('Year'); plt.ylabel('Number of applications')
    plt.legend()
    plt.savefig(council + 'reduced_coef_2.png')
    plt.show()
    return



## 1. Full Model
    
def run_full_bayesian_model_coefs(): 
    council_coef_dict = {}
    council_summary_dict = {}
    council_samples_dict = {}
    for council in council_reg_dict.keys():
        
            X_train = council_reg_dict[council][['SIMD',
                                  'housing_options', 'earnings', 'drug_deaths', 'pupil_attainment', 
                                  'fuel_poverty', 'drug_discharge', 'pop']]
        
            y_train = council_reg_dict[council]['Applications']
            n = len(y_train)
            data = {
                    'n':n,
                    'pop':X_train['pop'], 'pop_bar':np.mean(X_train['pop']), 'pop_sd':np.std(X_train['pop']),
                    'SIMD':X_train['SIMD'], 'SIMD_bar':np.mean(X_train['SIMD']), 'SIMD_sd':np.std(X_train['SIMD']),
                    'drug_deaths':X_train['drug_deaths'], 'drug_deaths_bar':np.mean(X_train['drug_deaths']), 'drug_deaths_sd':np.std(X_train['drug_deaths']),   
                    'earnings':X_train['earnings'], 'earnings_bar':np.mean(X_train['earnings']), 'earnings_sd':np.std(X_train['earnings']),
                    'pupil_attainment':X_train['pupil_attainment'], 'pupil_attainment_bar':np.mean(X_train['pupil_attainment']),'pupil_attainment_sd':np.std(X_train['pupil_attainment']),     
                    'fuel_poverty':X_train['fuel_poverty'], 'fuel_poverty_bar':np.mean(X_train['fuel_poverty']), 'fuel_poverty_sd':np.std(X_train['fuel_poverty']),
                    'drug_discharge':X_train['drug_discharge'], 'drug_discharge_bar':np.mean(X_train['drug_discharge']), 'drug_discharge_sd':np.std(X_train['drug_discharge']),
                    'housing_options':X_train['housing_options'], 'housing_options_bar':np.mean(X_train['housing_options']), 'housing_options_sd':np.std(X_train['drug_discharge']),
                    'y': y_train,  
                    }

            fit = bayesian_full_model.sampling(data=data, iter=10000, chains=2, warmup=5000, init=0,
                                               seed = 42, control=dict(max_treedepth=12))
            summary = fit.summary()
            summary_df = pd.DataFrame(summary['summary'], columns = summary['summary_colnames'],
                                      index = summary['summary_rownames'])
            results = fit.extract()
            results_dict = {}
            for key in list(results.keys()):
                results_dict[key] = [np.mean(results[key]), np.quantile(results[key], (0.1, 0.9))]
            
            predictions = CalcPredictions(results, council_reg_dict[council])
            PlotPredictions(predictions, council_reg_dict[council], council)
            
            council_coef_dict[council] = results_dict
            council_summary_dict[council] = summary_df
            council_samples_dict[council] = results
    return {'coef_dict':council_coef_dict, 'summary_dict':council_summary_dict, 
            'samples_dict':council_samples_dict}

bayesian_full_model = pystan.StanModel(model_code = bayesian_full_model_code)
bayesian_coef_full_model_results = run_full_bayesian_model_coefs()
    
## 2. Reduced Model

def run_reduced_bayesian_model_coefs(): 
    council_coef_dict = {}
    council_summary_dict = {}
    council_samples_dict = {}
    for council in council_reg_dict.keys():
        
            X_train = council_reg_dict[council][['SIMD',
                                  'housing_options', 'earnings', 'drug_deaths', 'pupil_attainment', 
                                  'fuel_poverty', 'drug_discharge', 'pop']]
        
            y_train = council_reg_dict[council]['Applications']
            n = len(y_train)
            data = {
                    'n':n,
                    'pop':X_train['pop'], 'pop_bar':np.mean(X_train['pop']), 'pop_sd':np.std(X_train['pop']),
                    'SIMD':X_train['SIMD'], 'SIMD_bar':np.mean(X_train['SIMD']), 'SIMD_sd':np.std(X_train['SIMD']),
                    'earnings':X_train['earnings'], 'earnings_bar':np.mean(X_train['earnings']), 'earnings_sd':np.std(X_train['earnings']),
                    'pupil_attainment':X_train['pupil_attainment'], 'pupil_attainment_bar':np.mean(X_train['pupil_attainment']),'pupil_attainment_sd':np.std(X_train['pupil_attainment']),     
                    'fuel_poverty':X_train['fuel_poverty'], 'fuel_poverty_bar':np.mean(X_train['fuel_poverty']), 'fuel_poverty_sd':np.std(X_train['fuel_poverty']),
                    'drug_discharge':X_train['drug_discharge'], 'drug_discharge_bar':np.mean(X_train['drug_discharge']), 'drug_discharge_sd':np.std(X_train['drug_discharge']),
                    'y': y_train,  
                    }

            fit = bayesian_reduced_model.sampling(data=data, iter=10000, chains=2, warmup=5000, init=0,
                                               seed = 42, control=dict(max_treedepth=12))
            summary = fit.summary()
            summary_df = pd.DataFrame(summary['summary'], columns = summary['summary_colnames'],
                                      index = summary['summary_rownames'])
            results = fit.extract()
            results_dict = {}
            for key in list(results.keys()):
                results_dict[key] = [np.mean(results[key]), np.quantile(results[key], (0.1, 0.9))]
            
            predictions = CalcPredictionsReduced(results, council_reg_dict[council])
            PlotPredictions(predictions, council_reg_dict[council], council)
            
            council_coef_dict[council] = results_dict
            council_summary_dict[council] = summary_df
            council_samples_dict[council] = results
    return {'coef_dict':council_coef_dict, 'summary_dict':council_summary_dict, 
            'samples_dict':council_samples_dict}

bayesian_coef_reduced_model_results = run_reduced_bayesian_model_coefs()

## 3. Baseline model
    
def run_baseline_bayesian_model_coefs(): 
    council_coef_dict = {}
    council_summary_dict = {}
    council_samples_dict = {}
    for council in council_reg_dict.keys():
        
            X_train = council_reg_dict[council][['SIMD',
                                  'housing_options', 'earnings', 'drug_deaths', 'pupil_attainment', 
                                  'fuel_poverty', 'drug_discharge', 'pop']]
        
            y_train = council_reg_dict[council]['Applications']
            n = len(y_train)
            data = {
                    'n':n,
                    'y': y_train,  
                    }

            fit = bayesian_baseline_model.sampling(data=data, iter=10000, chains=2, warmup=5000, init=0,
                                               seed = 42, control=dict(max_treedepth=12))
            summary = fit.summary()
            summary_df = pd.DataFrame(summary['summary'], columns = summary['summary_colnames'],
                                      index = summary['summary_rownames'])
            results = fit.extract()
            results_dict = {}
            for key in list(results.keys()):
                results_dict[key] = [np.mean(results[key]), np.quantile(results[key], (0.1, 0.9))]
            
            predictions = CalcPredictionsBaseline(results, council_reg_dict[council])
            PlotPredictions(predictions, council_reg_dict[council], council)
            
            council_coef_dict[council] = results_dict
            council_summary_dict[council] = summary_df
            council_samples_dict[council] = results
    return {'coef_dict':council_coef_dict, 'summary_dict':council_summary_dict, 
            'samples_dict':council_samples_dict}    

    

bayesian_coef_baseline_model_results = run_baseline_bayesian_model_coefs()


##################################################
          # 2. Results Analysis #
################################################## 

## 1. Compare RMSE of 3 models

# 1. Full model
council_rmse_df_full_mod = pd.DataFrame()
council_rmse_df_full_mod['Year'] = [2003, 2006, 2013, 2017, 2018]

for council in bayesian_validation_model_full_results['rmse_dicts'].keys():
    council_rmse_df_full_mod[council] = bayesian_validation_model_full_results['rmse_dicts'][council].values()
      
council_rmse_df_full_mod.iloc[:,1:].boxplot()    
plt.xticks(rotation = 90)

#Excluding Scotland
council_rmse_df_full_mod.iloc[:,1:-1].boxplot()    
plt.xticks(rotation = 90)
plt.ylim((0, 4500))
plt.title('Boxplot of rmse values obtained by full model for each council')
plt.ylabel('Number of applications')

np.mean(council_rmse_df_full_mod.iloc[:,1:-1].mean()) #367.5791555349033

## 2. Reduced model
council_rmse_df_reduced_mod = pd.DataFrame()
council_rmse_df_reduced_mod['Year'] = [2003, 2006, 2013, 2017, 2018]

for council in bayesian_validation_model_reduced_results['rmse_dicts'].keys():
    council_rmse_df_reduced_mod[council] = bayesian_validation_model_reduced_results['rmse_dicts'][council].values()

council_rmse_df_reduced_mod.iloc[:,1:].boxplot()
plt.xticks(rotation = 90)

#Excluding Scotland
council_rmse_df_reduced_mod.iloc[:,1:-1].boxplot()
plt.xticks(rotation = 90)
plt.ylim((0, 4500))
plt.title('Boxplot of rmse values obtained by reduced model for each council')
plt.ylabel('Number of applications')

np.mean(council_rmse_df_reduced_mod.iloc[:,1:-1].mean()) #368.80852309848854

## 3. Baseline model

council_rmse_df_baseline_mod = pd.DataFrame()
council_rmse_df_baseline_mod['Year'] = [2003, 2006, 2013, 2017, 2018]

for council in bayesian_validation_model_baseline_results['rmse_dicts'].keys():
    council_rmse_df_baseline_mod[council] = bayesian_validation_model_baseline_results['rmse_dicts'][council].values()


council_rmse_df_baseline_mod.iloc[:,1:].boxplot()
plt.xticks(rotation = 90)

#Excluding Scotland
council_rmse_df_baseline_mod.iloc[:,1:-1].boxplot()
plt.xticks(rotation = 90)
plt.ylim((0, 4500))
plt.title('Boxplot of rmse values obtained by baseline model for each council')
plt.ylabel('Number of applications')

np.mean(council_rmse_df_baseline_mod.iloc[:,1:-1].mean()) #379.1061910027917



## 2. Compare boxplots of coefficients of 3 models

# Create dataframes
alpha_df_full = pd.DataFrame()
for council in bayesian_coef_full_model_results['samples_dict'].keys():
    alpha_df_full[council] = bayesian_coef_full_model_results['samples_dict'][council]['alpha']
alpha_df_reduced = pd.DataFrame()
for council in bayesian_coef_reduced_model_results['samples_dict'].keys():
    alpha_df_reduced[council] = bayesian_coef_reduced_model_results['samples_dict'][council]['alpha']
alpha_df_baseline = pd.DataFrame()
for council in bayesian_coef_baseline_model_results['samples_dict'].keys():
    alpha_df_baseline[council] = bayesian_coef_baseline_model_results['samples_dict'][council]['alpha']    


beta0_df_full = pd.DataFrame()
for council in bayesian_coef_full_model_results['samples_dict'].keys():
    beta0_df_full[council] = bayesian_coef_full_model_results['samples_dict'][council]['beta0']
beta0_df_reduced = pd.DataFrame()
for council in bayesian_coef_reduced_model_results['samples_dict'].keys():
    beta0_df_reduced[council] = bayesian_coef_reduced_model_results['samples_dict'][council]['beta0']

beta1_df_full = pd.DataFrame()
for council in bayesian_coef_full_model_results['samples_dict'].keys():
    beta1_df_full[council] = bayesian_coef_full_model_results['samples_dict'][council]['beta1']



beta2_df_full = pd.DataFrame()
for council in bayesian_coef_full_model_results['samples_dict'].keys():
    beta2_df_full[council] = bayesian_coef_full_model_results['samples_dict'][council]['beta2']


beta2_df_reduced = pd.DataFrame()
for council in bayesian_coef_full_model_results['samples_dict'].keys():
    beta2_df_reduced[council] = bayesian_coef_reduced_model_results['samples_dict'][council]['beta2']



beta3_df_full = pd.DataFrame()
for council in bayesian_coef_full_model_results['samples_dict'].keys():
    beta3_df_full[council] = bayesian_coef_full_model_results['samples_dict'][council]['beta3']
beta3_df_reduced = pd.DataFrame()
for council in bayesian_coef_reduced_model_results['samples_dict'].keys():
    beta3_df_reduced[council] = bayesian_coef_reduced_model_results['samples_dict'][council]['beta3']


    
beta4_df_full = pd.DataFrame()
for council in bayesian_coef_full_model_results['samples_dict'].keys():
    beta4_df_full[council] = bayesian_coef_full_model_results['samples_dict'][council]['beta4']
beta4_df_reduced = pd.DataFrame()
for council in bayesian_coef_reduced_model_results['samples_dict'].keys():
    beta4_df_reduced[council] = bayesian_coef_reduced_model_results['samples_dict'][council]['beta4']


beta5_df_full = pd.DataFrame()
for council in bayesian_coef_full_model_results['samples_dict'].keys():
    beta5_df_full[council] = bayesian_coef_full_model_results['samples_dict'][council]['beta5']
beta5_df_reduced = pd.DataFrame()
for council in bayesian_coef_reduced_model_results['samples_dict'].keys():
    beta5_df_reduced[council] = bayesian_coef_reduced_model_results['samples_dict'][council]['beta5']


beta6_df_full = pd.DataFrame()
for council in bayesian_coef_full_model_results['samples_dict'].keys():
    beta6_df_full[council] = bayesian_coef_full_model_results['samples_dict'][council]['beta6']
beta6_df_reduced = pd.DataFrame()
for council in bayesian_coef_reduced_model_results['samples_dict'].keys():
    beta6_df_reduced[council] = bayesian_coef_reduced_model_results['samples_dict'][council]['beta6']


beta7_df_full = pd.DataFrame()
for council in bayesian_coef_full_model_results['samples_dict'].keys():
    beta7_df_full[council] = bayesian_coef_full_model_results['samples_dict'][council]['beta7']





# Plot coefficients
council_data_yrs['log_apps'] = np.log(council_data_yrs['Applications'])
council_data_yrs.iloc[1:-1,:].boxplot('log_apps', by = 'Council', showfliers=False)
plt.xticks(rotation = 90)
plt.ylabel('Log Number of Applications')
plt.title('Boxplots of the log number of applications in each council each \n year from 2003 to 2018')
plt.suptitle('') 
plt.ylim((4.5, 10))


simd_data.boxplot()
plt.xticks(rotation=90)
   
alpha_df_full.iloc[:,:-1].boxplot()
plt.hlines(y=0, xmin = 0, xmax = 33, color = 'red')
plt.xticks(rotation = 90)
plt.ylim((4.5,10))
plt.title('Full Model: Alpha - Intercept')
plt.show()


alpha_df_reduced.boxplot()
plt.hlines(y=0, xmin = 0, xmax = 33, color = 'red')
plt.xticks(rotation = 90)
plt.ylim((4.5,10))
plt.title('Reduced Model: Alpha - Intercept')
plt.show()    

alpha_df_baseline.iloc[:,:-1].boxplot()
plt.hlines(y=0, xmin = 0, xmax = 33, color = 'red')
plt.xticks(rotation = 90)
plt.ylim((4.5,10))
plt.title('Alpha - Intercept')
plt.show()   


beta0_df_full.iloc[:,:-1].boxplot()
plt.hlines(y=0, xmin = 0, xmax = 33, color = 'red')
plt.xticks(rotation = 90)
plt.ylim((-1,1))
plt.title('Full Model: Beta0 - SIMD')
plt.show()

beta0_df_reduced.iloc[:,:-1].boxplot()
plt.hlines(y=0, xmin = 0, xmax = 33, color = 'red')
plt.xticks(rotation = 90)
plt.ylim((-1,1))
plt.title('Reduced Beta0 - SIMD')
plt.show()

beta1_df_full.iloc[:,:-1].boxplot()
plt.hlines(y=0, xmin = 0, xmax = 33, color = 'red')
plt.xticks(rotation = 90)
plt.ylim((-1,1))
plt.title('Full Model: Beta1 - Drug Deaths')
plt.show()


beta2_df_full.iloc[:,:-1].boxplot()
plt.hlines(y=0, xmin = 0, xmax = 33, color = 'red')
plt.xticks(rotation = 90)
plt.ylim((-1,1))
plt.title('Full Model: Beta2 - Earnings')
plt.show()

beta2_df_reduced.iloc[:,:-1].boxplot()
plt.hlines(y=0, xmin = 0, xmax = 33, color = 'red')
plt.xticks(rotation = 90)
plt.ylim((-1,1))
plt.title('Reduced Model: Beta2 - Earnings')
plt.show()

beta3_df_full.iloc[:,:-1].boxplot()
plt.hlines(y=0, xmin = 0, xmax = 33, color = 'red')
plt.xticks(rotation = 90)
plt.ylim((-1,1))
plt.title('Full Model: Beta3 - Pupil Attainment')
plt.show()

beta3_df_reduced.iloc[:,:-1].boxplot()
plt.hlines(y=0, xmin = 0, xmax = 33, color = 'red')
plt.xticks(rotation = 90)
plt.ylim((-1,1))
plt.title('Reduced Model: Beta3 - Pupil Attainment')
plt.show()


beta4_df_full.iloc[:,:-1].boxplot()
plt.hlines(y=0, xmin = -1, xmax = 34, color = 'red')
plt.xticks(rotation = 90)
plt.ylim((-1,1))
plt.title('Full Model: Beta4 - Fuel Poverty')
plt.show()

beta4_df_reduced.iloc[:,:-1].boxplot()
plt.hlines(y=0, xmin = -1, xmax = 34, color = 'red')
plt.xticks(rotation = 90)
plt.ylim((-1,1))
plt.title('Reduced Beta4 - Fuel Poverty')
plt.show()


beta5_df_full.iloc[:,:-1].boxplot()
plt.hlines(y=0, xmin = -1, xmax = 34, color = 'red')
plt.xticks(rotation = 90)
plt.ylim((-1,1))
plt.title('Full Model: Beta5 - Population')
plt.show()

beta5_df_reduced.iloc[:,:-1].boxplot()
plt.hlines(y=0, xmin = -1, xmax = 34, color = 'red')
plt.xticks(rotation = 90)
plt.ylim((-1,1))
plt.title('Reduced Model: Beta5 - Population')
plt.show()

beta6_df_full.iloc[:,:-1].boxplot()
plt.hlines(y=0, xmin = -1, xmax = 34, color = 'red')
plt.xticks(rotation = 90)
plt.ylim((-1,1))
plt.title('Full Model: Beta6 - Drug Discharges')
plt.show()

beta6_df_reduced.iloc[:,:-1].boxplot()
plt.hlines(y=0, xmin = -1, xmax = 34, color = 'red')
plt.xticks(rotation = 90)
plt.ylim((-1,1))
plt.title('Reduced Model: Beta6 - Drug Discharges')
plt.show()

beta7_df_full.iloc[:,:-1].boxplot()
plt.hlines(y=0, xmin = -1, xmax = 34, color = 'red')
plt.xticks(rotation = 90)
plt.ylim((-1,1))
plt.title('Full Model: Beta7 - Housing Options')
plt.show()




plt.scatter(council_reg_dict['Glasgow City']['SIMD'], council_reg_dict['Glasgow City']['Applications'])
plt.scatter(council_reg_dict['Falkirk']['SIMD'], council_reg_dict['Falkirk']['Applications'])
plt.scatter(council_reg_dict['Shetland']['SIMD'], council_reg_dict['Shetland']['Applications'])

plt.scatter(council_reg_dict['Falkirk']['pupil_attainment'],council_reg_dict['Falkirk']['Applications'])
plt.scatter(council_reg_dict['North Ayrshire']['pupil_attainment'],council_reg_dict['North Ayrshire']['Applications'])
plt.scatter(council_reg_dict['North Ayrshire']['pupil_attainment'],council_reg_dict['North Ayrshire']['Applications'])


plt.scatter(council_reg_dict['Stirling']['earnings'], council_reg_dict['Stirling']['Applications'])
plt.scatter(council_reg_dict['East Dunbartonshire']['earnings'], council_reg_dict['East Dunbartonshire']['Applications'])
plt.scatter(council_reg_dict['Aberdeen City']['earnings'], council_reg_dict['Aberdeen City']['Applications'])




### Cluster Analysis

cluster_analysis_df = pd.DataFrame()
cluster_analysis_df['beta0'] = beta0_df_full.median(axis=0)
cluster_analysis_df['beta1'] = beta1_df_full.median(axis=0)
cluster_analysis_df['beta2'] = beta2_df_full.median(axis=0)
cluster_analysis_df['beta3'] = beta3_df_full.median(axis=0)
cluster_analysis_df['beta4'] = beta4_df_full.median(axis=0)
cluster_analysis_df['beta5'] = beta5_df_full.median(axis=0)
cluster_analysis_df['beta6'] = beta6_df_full.median(axis=0)
cluster_analysis_df['beta7'] = beta7_df_full.median(axis=0)

cluster_analysis_df = cluster_analysis_df.round(1)

cosine_similarity = cosine_similarity(cluster_analysis_df)
cosine_similarity_indices = cosine_similarity > 0.7
cosine_similarity_df = pd.DataFrame(cosine_similarity_indices, columns = cluster_analysis_df.index,
                                    index = cluster_analysis_df.index)



# PCA fun

from sklearn.decomposition import PCA
import matplotlib.cm as cm
from matplotlib.colors import Normalize

pca = PCA(n_components = 2)
pca.fit(cosine_similarity)
df_pca = pca.transform(cosine_similarity)

colors = np.arctan2(df_pca[:,1], df_pca[:, 0])
norm = Normalize()
norm.autoscale(colors)
colormap = cm.inferno

plt.quiver(np.repeat(0, 33), np.repeat(0, 33), df_pca[:, 0], df_pca[:,1], color=colormap(norm(colors)), scale=1,
           angles='xy', scale_units='xy', label = cosine_similarity_df.index)
plt.ylim((-3, 3))
plt.xlim((-3, 3))
plt.title('PCA Council Vectors')



pca_df = pd.DataFrame(df_pca)
pca_df.index = cosine_similarity_df.index
pca_df[(pca_df[0]>0) & (pca_df[0]<2) & (pca_df[1]>1.3)] #Orange
pca_df[(pca_df[0]<-0.8) & (pca_df[1]<0.2)].index #Black
np.array(pca_df[(pca_df[0]<-0.5) & (pca_df[1]<2.2) & (pca_df[1]>0.3)].index) #Yellow
np.array(pca_df[(pca_df[0]<0) & (pca_df[0]>-0.8) & (pca_df[1]<-1)].index) # Dark Purple
np.array(pca_df[(pca_df[0]>0) & (pca_df[1]<1.5) & (pca_df[1]>-1)].index) #Red
np.array(pca_df[(pca_df[0]>0) & (pca_df[1]<-1)].index) #Light purple

#Yellow: 'Dumfries & Galloway', 'Eilean Siar', 'North Lanarkshire', 'Scottish Borders', 'South Ayrshire', 'South Lanarkshire'
#Yellow - Orange : West Dunbartonshire
#Orange: Argyll & Bute, Falkirk, West Lothian
# Red: Aberdeenshire', 'Angus', 'Clackmannanshire',  Falkirk, 'Fife', 'Midlothian', 'Moray', 'Perth & Kinross', 'Renfrewshire'
# Light Purple: 'East Ayrshire', 'East Renfrewshire', 'Highland', 'Orkney'
#Dark Purple: Aberdeen City', 'Edinburgh', 'Stirling'
#Black: 'Dundee City', 'East Dunbartonshire', 'East Lothian', 'Glasgow City', 'North Ayrshire', 'Shetland'





quiver(np.x1, np.y1, np.x2-np.x1, np.y2-np.y1, angles='xy', scale_units='xy', scale=1)



