#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 16:51:16 2019

@author: intimantripp
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
from math import isnan


##################################################
           # Load Main Data In #
##################################################  
           

path = '/Users/intimantripp/Desktop/Documents/University of Edinburgh/Summer Projects/Project 2/Data/'
os.chdir(path)

natl_data = pd.read_csv(path + 'natl.yr.mth.data.csv')
council_data = pd.read_csv(path + 'council.mth.data.csv')
council_data['YearMonth'] = council_data['Year'].map(str) + '-' + council_data['Month'].map(str)
council_data['YearMonth'] = council_data['YearMonth'].astype('datetime64[D]')

natl_data['YearMonth'] = natl_data['Year'].map(str) + '-' +  natl_data['Month'].map(str)
natl_data['YearMonth'] = natl_data['YearMonth'].astype('datetime64[D]')
natl_data['With_Children'] = (natl_data['Single.Male.Parent'] + natl_data['Single.Female.Parent'] +
                              natl_data['Couple.and.Children'] + natl_data['Other.and.Children'])


natl_data_yrs = natl_data.groupby(['Year'], as_index = False).agg(np.sum)
natl_data_yrs['pop'] = natl_data.groupby(['Year'], as_index = False).agg(np.mean)['pop']
natl_data_yrs['With_Children']/natl_data_yrs['Homeless.app']


##################################################
           # Load Secondary Data In #
##################################################  

ethnic_data = pd.read_csv(path + 'ethnic_data.csv')


############## FUNCTIONS #####################

def draw_boxplot(data, edge_color, fill_color):
    bp = ax.boxplot(data, patch_artist=True, showfliers = False)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)
        
    for median in bp['medians']:
        median.set(color='#0D1315', linewidth=1)

    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)

def HomelessCountEst(observation):
    
    count = 0
    categories = {'Single.Male':1,
                  'Single.Female':1,
                  'Single.Male.Parent':2, 'Single.Female.Parent':2, 'Couple.Only':2, 
                  'Couple.and.Children':3, 'Other':2, 'Other.and.Children':2}
    for category in list(categories.keys()):
        count = count + observation[category]*categories[category]
    return count

##################################################
           # Challenge 1 #
##################################################  

## How many homeless?

natl_data['Homeless.app'].iloc[-1]
natl_data['Homeless.assess'].iloc[-1]


HomelessCountEst(natl_data.iloc[-1])    


## Demographic breakdown

natl_data.iloc[-1][['Single.Male', 'Single.Female', 'Single.Male.Parent', 'Single.Female.Parent', 'Couple.Only',
              'Couple.and.Children', 'Other', 'Other.and.Children']].plot.bar(
              title = 'Demographic breakdown of Applications in March 2019')

natl_data.iloc[-2][['Single.Male', 'Single.Female', 'Single.Male.Parent', 'Single.Female.Parent', 'Couple.Only',
              'Couple.and.Children', 'Other', 'Other.and.Children']].plot.bar(
              title = 'Demographic breakdown of Applications in February 2019')    

natl_data.iloc[-3][['Single.Male', 'Single.Female', 'Single.Male.Parent', 'Single.Female.Parent', 'Couple.Only',
              'Couple.and.Children', 'Other', 'Other.and.Children']].plot.bar(
              title = 'Demographic breakdown of Applications in January 2019')    

natl_data[['Single.Male', 'Single.Female', 'Single.Male.Parent', 'Single.Female.Parent', 'Couple.Only',
              'Couple.and.Children', 'Other', 'Other.and.Children']].boxplot(rot = 45)


    
natl_data[natl_data['Year'] >= 2010][['Single.Male', 'Single.Female', 'Single.Male.Parent', 'Single.Female.Parent', 'Couple.Only',
              'Couple.and.Children', 'Other', 'Other.and.Children']].boxplot(rot = 45)

natl_data[natl_data['Year'] < 2010][['Single.Male', 'Single.Female', 'Single.Male.Parent', 'Single.Female.Parent', 'Couple.Only',
              'Couple.and.Children', 'Other', 'Other.and.Children']].boxplot(rot = 45)


demographic_list1 = ['Single.Male', 'Single.Female', 'Single.Male.Parent', 'Single.Female.Parent', 'Couple.Only',
              'Couple.and.Children', 'Other', 'Other.and.Children', 'With_Children']
demographic_xlabs1 = ['Single Male', 'Single Female', 'Single Male Parent', 'Single Female Parent', 'Couple Only',
              'Couple and Children', 'Other', 'Other and Children', 'Household with Children']
temp_dict = {key:natl_data_yrs[natl_data_yrs['Year']<2019][key] for key in demographic_list1}
keys, data = temp_dict.keys(), temp_dict.values()
fig, ax = plt.subplots()
draw_boxplot(data, '#7570b3', '#1f77b4')
plt.xticks(list(np.arange(1, 10)), demographic_xlabs1, rotation=45)
plt.ylabel('Number of Applications')
plt.title('Boxplot of number of applications made each year by different groups')
plt.show()


demographic_list2 = list(ethnic_data.columns[2:11])
temp_dict = {key:ethnic_data[key] for key in demographic_list2}
keys, data = temp_dict.keys(), temp_dict.values()

fig, ax = plt.subplots()
draw_boxplot(data, '#7570b3', '#1f77b4')
plt.xticks(list(np.arange(1, 9)), demographic_list2, rotation=45)
plt.ylabel('Number of Applications')
plt.title('Boxplot of number of applications made by different ethnic groups each  month (excluding White: Scottish)')
plt.show()

ethnic_data.iloc[:,2:11].boxplot(rot = 45)

ethnic_data['White: Scottish']/ethnic_data['Total']
ethnic_data['White: Other British']/ethnic_data['Total']


for ethnic in ethnic_data.columns[1:11]:
    ethnic_data[ethnic + ' ratio'] = ethnic_data[ethnic]/ethnic_data['Total']
    
ethnic_data.plot(x = 'Year', y = list(ethnic_data.columns[13:]))


# How many classified as repeat homeless?

plt.plot(natl_data['YearMonth'], natl_data['Assess.repeats']/1000, color = 'blue')
plt.plot(natl_data['YearMonth'], natl_data['Homeless.app']/10000, color = 'red')

plt.plot(natl_data_yrs['Year'], natl_data_yrs['Assess.repeats']/1000, color = 'blue')
plt.plot(natl_data_yrs['Year'], natl_data_yrs['Homeless.app']/10000, color = 'red')

plt.plot(natl_data['YearMonth'], natl_data['Assess.repeats']/natl_data['Homeless.app'])
plt.scatter(natl_data[natl_data['Month']==12]['YearMonth'], 
            natl_data[natl_data['Month']==12]['Assess.repeats']/natl_data[natl_data['Month']==12]['Homeless.app'])

##################################################
 # 2. Long term trends and regional variations #
##################################################
 

### Long term trends ### 

## National ##

## Plot of how homeless applications evolve over time

plt.plot(natl_data['YearMonth'], natl_data['Homeless.app'], label = 'Applications')
plt.xlabel('Year')
plt.ylabel('Number of Applicants')
plt.title('Number of Applicants applying as homeless in Scotland each month')
plt.show()

plt.plot(natl_data_yrs[natl_data_yrs['Year']<2019]['Year'], 
         natl_data_yrs[natl_data_yrs['Year']<2019]['Homeless.app'], label = 'Applications')
plt.plot(natl_data_yrs[natl_data_yrs['Year']<2019]['Year'], 
         natl_data_yrs[natl_data_yrs['Year']<2019]['Homeless.assess'], label = 'Assessments')
plt.xlabel('Year')
plt.legend()
plt.title('Number of households applying and assessed as homeless in Scotland each year')

#Maths stuff

natl_data['ass_rat'] = natl_data['Homeless.assess']/natl_data['Homeless.app']
natl_data['ass_rat'].quantile((0.1, 0.9))
np.std(natl_data['ass_rat'])
np.median(natl_data['ass_rat'])


# The December anomoly
plt.clf()
fig, ax = plt.subplots()
plot1, = plt.plot(natl_data['YearMonth'], natl_data['Homeless.app'])
plot2 = plt.scatter(natl_data[natl_data['Month']==12]['YearMonth'], natl_data[natl_data['Month']==12]['Homeless.app'],
            color = 'red', marker = 'x')
plot3 = plt.scatter(natl_data[natl_data['Month']==11]['YearMonth'], natl_data[natl_data['Month']==11]['Homeless.app'],
            color = 'green', marker = 'x')
plt.legend([plot1, plot2, plot3], ['Applications', 'Applications in Dec', 'Applications in Nov']);
plt.title('Number of homeless applications in Scotland each month')
plt.xlabel('Year')
plt.ylabel('Number of applications')

# Couples delaying until January
# Services swing into action November, December


natl_data.groupby('Month', as_index = False).agg(np.mean).plot.bar(x = 'Month', y = 'Homeless.app')

test = {month:natl_data[natl_data['Month']==month]['Homeless.app'] for month in list(np.unique(natl_data['Month']))}
months, data = test.keys(), test.values()


def draw_boxplot(data, edge_color, fill_color):
    bp = ax.boxplot(data, patch_artist=True, showfliers = False)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)
        
    for median in bp['medians']:
        median.set(color='#0D1315', linewidth=1)

    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)
fig, ax = plt.subplots()
draw_boxplot(data, '#7570b3', '#1f77b4')
plt.xticks(list(test.keys()), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.xlabel('Month')
plt.ylabel('Number of Homeless Applications')
plt.title('Boxplots of number of homeless Applications each month')
plt.show()



# Examining how homeless population evolves
plt.plot(natl_data['YearMonth'], natl_data['Homeless.app'], label = 'Households applying for homeless status')
plt.plot(natl_data['YearMonth'], natl_data['Homeless.assess'], label = 'Households assessed as homeless')
plt.legend()
plt.plot(natl_data['YearMonth'], natl_data['Homeless.app']-natl_data['Homeless.assess'])

plt.plot(natl_data_yrs['Year'], natl_data_yrs['Homeless.app'], label = 'Number of Households Applying')
plt.plot(natl_data_yrs['Year'], natl_data_yrs['Homeless.assess'], label = 'Number of Households Assessed as Homeless')
plt.legend()
np.std(natl_data['Homeless.app']-natl_data['Homeless.assess'])


plt.plot(natl_data_yrs['Year'][(natl_data_yrs['Year'] > 2002) & (natl_data_yrs['Year'] < 2019)], 
         natl_data_yrs['Homeless.assess'][(natl_data_yrs['Year'] > 2002) & (natl_data_yrs['Year'] < 2019)]
         /natl_data_yrs['Homeless.app'][(natl_data_yrs['Year'] > 2002) & (natl_data_yrs['Year'] < 2019)])
plt.ylim((0.3, 1))
plt.title('Evolution in the proportion of applications approved each year')
plt.ylabel('Proportion')
plt.xlabel('Year')

# Change in the population
np.std(natl_data[natl_data['Year']< 2019]['pop'])

# Homeless applications prior to Homeless options
np.mean(natl_data[natl_data['Year']<2010]['Homeless.app'])
np.std(natl_data[natl_data['Year']<2010]['Homeless.app'])

# Subsequent to homeless options
np.mean(natl_data[natl_data['Year']>2010]['Homeless.app'])
np.std(natl_data[natl_data['Year']>2010]['Homeless.app'])

np.mean(natl_data['Homeless.app']-natl_data['Homeless.assess'])


plt.plot(natl_data['YearMonth'], natl_data['Homeless.app'], label = 'Applications')
plt.plot(natl_data['YearMonth'], natl_data['Homeless.assess'], label = 'Assessments')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Quantity')
plt.title('Applications made and applications assessed each month')

## Plot of proportion of accepted homeless with upper and lower extremes marked.
plt.plot(natl_data['YearMonth'], natl_data['Homeless.assess']/natl_data['Homeless.app'],
         label = 'Proportion of applications accepted')
plt.scatter(natl_data[natl_data['Month']==1]['YearMonth'], 
            natl_data[natl_data['Month']==1]['Homeless.assess']/natl_data[natl_data['Month']==1]['Homeless.app'], 
            color='red', label = 'Observations in January', marker = 'x')
plt.scatter(natl_data[natl_data['Month']==12]['YearMonth'], 
            natl_data[natl_data['Month']==12]['Homeless.assess']/natl_data[natl_data['Month']==12]['Homeless.app'], 
            color='green', label = 'Observations in December', marker = 'x')
plt.legend(loc='upper left')
plt.title('Proportion of applications that are assessed as homeless each month')

temp_dict = {month:natl_data[natl_data['Month']==month]['ass_rat'] for month in list(np.unique(natl_data['Month']))}
keys, data = temp_dict.keys(), temp_dict.values()

fig, ax = plt.subplots()
draw_boxplot(data, '#7570b3', '#1f77b4')
plt.xticks(list(test.keys()), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.xlabel('Month')
plt.ylabel('Proportion')
plt.title('Boxplots of proportions of applications assessed each month')
plt.show()

natl_data[natl_data['Year']==2017]['ass_rat']
#Examine changes in proportions of different groups

natl_data['single_male_rat'] = natl_data['Single.Male']/natl_data['Homeless.app']
natl_data['single_fem_rat'] = natl_data['Single.Female']/natl_data['Homeless.app']
natl_data['male_par_rat'] = natl_data['Single.Male.Parent']/natl_data['Homeless.app']
natl_data['female_par_rat'] = natl_data['Single.Female.Parent']/natl_data['Homeless.app']
natl_data['couple_rat'] = natl_data['Couple.Only']/natl_data['Homeless.app']
natl_data['couple_par_rat'] = natl_data['Couple.and.Children']/natl_data['Homeless.app']
natl_data['other_rat'] = natl_data['Other']/natl_data['Homeless.app']
natl_data['other_child_rat'] = natl_data['Other.and.Children']/natl_data['Homeless.app']

plot_headings = list(natl_data.columns)[17:]

for heading in plot_headings:
    plt.plot(natl_data['YearMonth'], natl_data[heading])
plt.scatter(natl_data[natl_data['Month']==12]['YearMonth'], natl_data[natl_data['Month']==12]['single_male_rat'],
            marker = 'x', color = 'red')
plt.scatter(natl_data[natl_data['Month']==12]['YearMonth'], natl_data[natl_data['Month']==12]['female_par_rat'],
            marker = 'x', color = 'green')
plt.xlabel('Year')
plt.ylabel('Proportion')
plt.legend(['Single Males', 'Single Females', 'Single Male Parents', 'Single Female Parents', 'Couple', 
            'Couples with children', 'Others', 'Others with children', 'Single Males - Dec', 
            'Single Female Parents - Dec'], bbox_to_anchor=(1, 1.05))
plt.title('Proportion of applicants in different groups each month')
    
natl_data['rough_3_rat'] = natl_data['rough.3mths']/natl_data['Homeless.app']
natl_data['rough_night_rat'] = natl_data['rough.last.nite']/natl_data['Homeless.app']
    
plt.plot(natl_data['YearMonth'], natl_data['rough_3_rat'], label = 'Rough within 3 months')
plt.plot(natl_data['YearMonth'], natl_data['rough_night_rat'], label = 'Rough last night')    
plt.scatter(natl_data[natl_data['Month']==3]['YearMonth'], natl_data[natl_data['Month']==3]['rough_3_rat'],
            color = 'red', marker = 'x')
plt.scatter(natl_data[natl_data['Month']==3]['YearMonth'], natl_data[natl_data['Month']==3]['rough_night_rat'],
            color = 'green', marker = 'x')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Proportion')
plt.title('Evolution of proportion of applicants that slept rough')

    
temp_dict = {month:natl_data[natl_data['Month']==month]['rough.3mths'] for month in list(np.unique(natl_data['Month']))}
months, data = temp_dict.keys(), temp_dict.values()

fig, ax = plt.subplots()
draw_boxplot(data, '#7570b3', '#1f77b4')
plt.xticks(list(test.keys()), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.xlabel('Month')
plt.ylabel('Number of Rough sleeping homeless applications')
plt.title('Boxplots of number of rough sleeping homeless applications each month from years 2002 to 2019')
plt.show()


temp_dict = {month:natl_data[natl_data['Month']==month]['rough.last.nite'] for month in list(np.unique(natl_data['Month']))}
months, data = temp_dict.keys(), temp_dict.values()

fig, ax = plt.subplots()
draw_boxplot(data, '#7570b3', '#1f77b4')
plt.xticks(list(test.keys()), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.xlabel('Month')
plt.ylabel('Number of Rough sleeping homeless applications')
plt.title('Boxplots of number of rough sleeping homeless applications each month from years 2002 to 2019')
plt.show()



#Now examining different councils

os.chdir('/Users/intimantripp/Desktop/Documents/University of Edinburgh/Summer Projects/Project 2/figures')

## Manipulate council data

council_data_yrs = council_data.groupby(['Council', 'Year'], as_index = False).agg(sum)
council_data_yrs['pop'] = council_data.groupby(['Council', 'Year'], as_index = False).agg(np.mean)['pop']

council_list = list(np.unique(council_data_yrs['Council']))
council_dict = {council:council_data[council_data['Council'] ==council] for council in council_list }
council_dict_yrs = {council:council_data_yrs[council_data_yrs['Council']==council] for council in council_list}
council_dict_yrs['Scotland'] = council_data_yrs.groupby(['Year'], as_index = False).agg(np.sum)
council_list = list(council_dict_yrs.keys())

## Create new variables
council_data['app_ratio'] = council_data['Applications']/council_data['pop']

for council in council_list:
    council_dict_yrs[council]['app_ratio'] = council_dict_yrs[council]['Applications']/council_dict_yrs[council]['pop']
    council_dict_yrs[council]['ass_ratio'] = council_dict_yrs[council]['Assessments']/council_dict_yrs[council]['Applications']


for council in council_list:
    council_dict_yrs[council]['app_ratio'] = council_dict_yrs[council]['app_ratio'].fillna(0)

temp_dict = {council:council_dict_yrs[council]['app_ratio'].iloc[:-1] for council in council_list}

councils, data = temp_dict.keys(), temp_dict.values()

fig, ax = plt.subplots(figsize=(15, 14))
draw_boxplot(data, '#7570b3', '#1f77b4')
plt.xticks(list(range(1, 33)), list(councils), rotation=90)
plt.xlabel('Council')
plt.ylabel('Proportion')
plt.title('Boxplots of number of applications in each council each year per population')
plt.savefig('council_apprates_boxplots.pdf')
plt.show()


temp_dict = {council:council_dict_yrs[council]['ass_ratio'].iloc[:-1] for council in council_list}
councils, data = temp_dict.keys(), temp_dict.values()

fig, ax = plt.subplots(figsize=(15, 14))
draw_boxplot(data, '#7570b3', '#1f77b4')
plt.xticks(list(range(1, 33)), list(councils), rotation=90)
plt.xlabel('Council')
plt.ylabel('Proportion')
plt.title('Boxplots of proportion of applications assessed as homeless')
plt.savefig('council_assrates_boxplots.pdf')
plt.show()

plt.plot(council_dict_yrs['Edinburgh']['Year'], council_dict_yrs['Edinburgh']['ass_ratio'])

plt.plot(council_dict_yrs['Edinburgh']['Year'], council_dict_yrs['Edinburgh']['Applications'])
plt.plot(council_dict_yrs['Edinburgh']['Year'], council_dict_yrs['Edinburgh']['Assessments'])

##Populations of different councils

for council in council_list:
    plt.plot(council_dict[council]['YearMonth'], council_dict[council]['pop'])
    plt.legend(council_list, bbox_to_anchor=(1, 1.05))
plt.show()


temp_dict = {council: council_dict[council]['pop'][:-3] for council in list(council_dict.keys())}
councils, data = temp_dict.keys(), temp_dict.values()
fig, ax = plt.subplots()
draw_boxplot(data, '#7570b3', '#1f77b4')
plt.xticks(list(range(1, 33)), list(council_dict.keys()))
plt.show()

##################################################
         # 3. Seasonal Variations #
##################################################

natl_data_months = natl_data.groupby(['Month'], as_index = False).agg(np.mean)

plt.plot(natl_data['YearMonth'],natl_data['rough.3mths'],
         label = '3 months')
plt.scatter(natl_data[natl_data['Month']==12]['YearMonth'],
            natl_data[natl_data['Month']==12]['rough.3mths'], marker = 'x', color = 'red', 
            label = '3 months - Dec')
plt.plot(natl_data['YearMonth'], natl_data['rough.last.nite'], label = 'Last night')
plt.scatter(natl_data[natl_data['Month']==12]['YearMonth'], 
            natl_data[natl_data['Month']==12]['rough.last.nite'], label = 'Last night - Dec', 
            marker = 'x', color = 'blue')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Number of applicants')
plt.title('Number of applications received from individuals that had \n slept rough in the last 3 months and the night before applying')

plt.plot(natl_data['YearMonth'], natl_data['rough_3_rat'], label = '3 months')
plt.plot(natl_data['YearMonth'], natl_data['rough_night_rat'], label = 'Last night')
plt.xlabel('Year')
plt.ylabel('Proportion')
plt.title('Ratio of applicants that slept rough in the last 3 months \n and the night before applying')
plt.legend()

#Slept rough in last 3 months#
temp_dict = {month:natl_data[natl_data['Month']==month]['rough.3mths'] for month in list(np.unique(natl_data_months['Month']))}
months, data = temp_dict.keys(), temp_dict.values()
fig, ax = plt.subplots()
draw_boxplot(data, '#7570b3', '#1f77b4')
plt.xticks(list(temp_dict.keys()), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.xlabel('Month')
plt.ylabel('Number of Applicants')
plt.title('Boxplots of number of homeless applicants \n that indicated sleeping rough in past 3 months')


## Slept rough the night before#
temp_dict = {month:natl_data[natl_data['Month']==month]['rough.last.nite'] for month in list(np.unique(natl_data_months['Month']))}
months, data = temp_dict.keys(), temp_dict.values()
fig, ax = plt.subplots()
draw_boxplot(data, '#7570b3', '#1f77b4')
plt.xticks(list(temp_dict.keys()), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.xlabel('Month')
plt.ylabel('Number of Applicants')
plt.title('Boxplots of numbers of homeless applicants that \n indicated sleeping rough night before applying each month')

# Rough Night 3 months ratio
temp_dict = {month:natl_data[natl_data['Month']==month]['rough_3_rat'] for month in list(np.unique(natl_data_months['Month']))}
months, data = temp_dict.keys(), temp_dict.values()
fig, ax = plt.subplots()
draw_boxplot(data, '#7570b3', '#1f77b4')
plt.xticks(list(temp_dict.keys()), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.xlabel('Month')
plt.ylabel('Proportion')
plt.title('Boxplots of proportion of homeless applicants that \n slept rough in last 3 months')

# Rough last night ratio
temp_dict = {month:natl_data[natl_data['Month']==month]['rough_night_rat'] for month in list(np.unique(natl_data_months['Month']))}
months, data = temp_dict.keys(), temp_dict.values()
fig, ax = plt.subplots()
draw_boxplot(data, '#7570b3', '#1f77b4')
plt.xticks(list(temp_dict.keys()), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.xlabel('Month')
plt.ylabel('Proportion')
plt.title('Boxplots of proportion of homeless applicants that \n slept rough the night before applying')


##################################################
 # 4. Deviations between Applications and assessments in council areas #
##################################################

for council in list(council_dict_yrs.keys()):
    plt.plot(council_dict_yrs[council]['Year'], council_dict_yrs[council]['ass_ratio'], label = council)
    plt.ylim([0,1])
plt.legend(bbox_to_anchor=(1, 1.05)) 
plt.ylabel('Proportion')
plt.xlabel('Year')
plt.title('Evolution of proportion of applications that are assessed as homeless for each council')
plt.show()


for council in list(council_dict_yrs.keys()):
    plt.plot(council_dict_yrs[council]['Year'], council_dict_yrs[council]['app_ratio'], label = council)
    plt.ylim([0,0.03])
plt.legend(bbox_to_anchor=(1, 1.05)) 
plt.ylabel('Proportion')
plt.xlabel('Year')
plt.title('Evolution of proportion of population that apply for homeless status')
plt.show()



##################################################
     # 6. Effect of holiday letting #
##################################################

#Import airbnb data
     
airbnb_data = pd.read_csv(path + 'listings.csv')
airbnb_data['first_review'] = airbnb_data['first_review'].astype('datetime64')

airbnb_month_counts = airbnb_data['first_review'].groupby([airbnb_data.first_review.dt.year, airbnb_data.first_review.dt.month]).agg({'count'})
airbnb_month_counts['cum_sum'] = airbnb_month_counts.cumsum()

a = pd.DataFrame(airbnb_month_counts.index.tolist(), columns = ['year', 'month'])
a['day'] = 1
airbnb_month_counts = airbnb_month_counts.set_index(a.index)
airbnb_month_counts['YearMonth'] = a['year'].map(int).map(str) + '-' + a['month'].map(int).map(str)
airbnb_month_counts['YearMonth'] = airbnb_month_counts['YearMonth'].astype('datetime64[D]')

#Edinburgh dataframe
edi_data = council_dict['Edinburgh']


edi_data.plot(y = 'Applications',
              x = 'YearMonth')
airbnb_month_counts.plot(x = 'YearMonth', y = 'cum_sum')

