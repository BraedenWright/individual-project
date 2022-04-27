# Basics
import numpy as np
import pandas as pd
import os
import scipy.stats as stats
from pydataset import data
from scipy import math
import datetime
from datetime import datetime


def wrangle_labor():
    '''
    This function pulls pulls the labor.csv into a dataframe, then cleans the data by renaming columns, then feature engineers additional
    columns to work with.  Also set the index of the new .csv to a datetime index, then produces a new .csv of the cleaned data.
    '''
    
    filename = 'cleaned_labor.csv'
    
    if os.path.exists(filename):
        print('Reading cleaned data from csv file...')
        return pd.read_csv(filename)
    
    # aquire basic data
    df = pd.read_csv('labor.csv')
    
    # rename columns
    df = df.rename(columns = {
                          'Time.Month':'month_num', 
                          'Time.Month Name':'month', 
                          'Time.Year':'year',
                          'Data.Civilian Noninstitutional Population.Asian':'asian_american_cnp',
                          'Data.Civilian Noninstitutional Population.Black or African American':'african_american_cnp', 
                          'Data.Civilian Noninstitutional Population.White':'white_cnp',
                          'Data.Not In Labor Force.Asian':'asian_american_not_in_labor_force',
                          'Data.Not In Labor Force.Black or African American':'african_american_not_in_labor_force',
                          'Data.Not In Labor Force.White':'white_not_in_labor_force', 
                          'Data.Civilian Labor Force.Asian.Counts':'asian_american_civilian_labor_force', 
                          'Data.Civilian Labor Force.Asian.Participation Rate':'asian_american_clf_rate',
                          'Data.Employed.Asian.Counts':'asian_american_employed',  
                          'Data.Unemployed.Asian.Counts':'asian_american_unemployed',
                          'Data.Unemployed.Asian.Unemployment Rate':'asian_american_unemployment_rate', 
                          'Data.Civilian Labor Force.Black or African American.Counts.All':'all_african_american_civilian_labor_force',
                          'Data.Civilian Labor Force.Black or African American.Counts.Men':'male_african_american_civilian_labor_force',
                          'Data.Civilian Labor Force.Black or African American.Counts.Women':'female_african_american_civilian_labor_force',
                          'Data.Civilian Labor Force.Black or African American.Participation Rate.All':'all_african_american_clf_rate', 
                          'Data.Civilian Labor Force.Black or African American.Participation Rate.Men':'male_african_american_clf_rate', 
                          'Data.Civilian Labor Force.Black or African American.Participation Rate.Women':'female_african_american_clf_rate',
                          'Data.Civilian Labor Force.White.Counts.All':'all_white_civilian_labor_force', 
                          'Data.Civilian Labor Force.White.Counts.Men':'male_white_civilian_labor_force', 
                          'Data.Civilian Labor Force.White.Counts.Women':'female_white_civilian_labor_force',
                          'Data.Civilian Labor Force.White.Participation Rate.All':'all_white_clf_rate', 
                          'Data.Civilian Labor Force.White.Participation Rate.Men':'male_white_clf_rate',
                          'Data.Civilian Labor Force.White.Participation Rate.Women':'female_white_clf_rate',
                          'Data.Employed.Black or African American.Counts.All':'all_african_american_employed',
                          'Data.Employed.Black or African American.Counts.Men':'male_african_american_employed', 
                          'Data.Employed.Black or African American.Counts.Women':'female_african_american_employed', 
                          'Data.Employed.Black or African American.Employment-Population Ratio.All':'all_african_american_employment_ratio',
                          'Data.Employed.Black or African American.Employment-Population Ratio.Men':'male_african_american_employment_ratio', 
                          'Data.Employed.Black or African American.Employment-Population Ratio.Women':'female_african_american_employment_ratio', 
                          'Data.Employed.White.Counts.All':'all_white_employed',
                          'Data.Employed.White.Counts.Men':'male_white_employed', 
                          'Data.Employed.White.Counts.Women':'female_white_employed',
                          'Data.Employed.White.Employment-Population Ratio.All':'employed_all_white_employment_ratio',
                          'Data.Employed.White.Employment-Population Ratio.Men':'employed_male_white_employment_ratio',
                          'Data.Employed.White.Employment-Population Ratio.Women':'employed_female_white_employment_ratio', 
                          'Data.Unemployed.Black or African American.Counts.All':'all_african_american_unemployed', 
                          'Data.Unemployed.Black or African American.Counts.Men':'male_african_american_unemployed',
                          'Data.Unemployed.Black or African American.Counts.Women':'female_african_american_unemployed', 
                          'Data.Unemployed.Black or African American.Unemployment Rate.All':'all_african_american_unemployment_rate', 
                          'Data.Unemployed.Black or African American.Unemployment Rate.Men':'male_african_american_unemployment_rate',
                          'Data.Unemployed.Black or African American.Unemployment Rate.Women':'female_african_american_unemployment_rate', 
                          'Data.Unemployed.White.Counts.All':'all_white_unemployed',
                          'Data.Unemployed.White.Counts.Men':'male_white_unemployed',
                          'Data.Unemployed.White.Counts.Women':'female_white_unemployed',
                          'Data.Unemployed.White.Unemployment Rate.All':'all_white_unemployment_rate', 
                          'Data.Unemployed.White.Unemployment Rate.Men':'male_white_unemployment_rate', 
                          'Data.Unemployed.White.Unemployment Rate.Women':'female_white_unemployment_rate'})
    
    # Feature Engineer
    df['total_civ_non_population'] = df['asian_american_cnp'] + df['african_american_cnp'] + df['white_cnp']

    df['total_not_in_labor_force'] = df['asian_american_not_in_labor_force'] + df['african_american_not_in_labor_force'] + df['white_not_in_labor_force']

    df['avg_clf_rate'] = round(((df['asian_american_clf_rate'] + df['all_african_american_clf_rate'] + df['all_white_clf_rate']) / 3), 2)
    
    df['total_unemployed_pop'] = df['asian_american_unemployed'] + df['all_african_american_unemployed'] + df['all_white_unemployed']

    df['avg_female_clf_rate'] = round(((df['female_african_american_clf_rate'] + df['female_white_clf_rate']) / 2), 2)

    df['avg_male_clf_rate'] = round(((df['male_african_american_clf_rate'] + df['male_white_clf_rate']) / 2), 2)

    df['avg_female_unemployment_rate'] = round(((df['female_african_american_unemployment_rate'] + df['female_white_unemployment_rate']) / 2), 2)

    df['avg_male_unemployment_rate'] = round(((df['male_african_american_unemployment_rate'] + df['male_white_unemployment_rate']) / 2), 2)

    df['avg_total_unemployment_rate'] = round(((df['asian_american_unemployment_rate'] + df['all_african_american_unemployment_rate'] + df['all_white_unemployment_rate']) / 3), 2)

    df['unemployment_bin'] = pd.qcut(df.avg_total_unemployment_rate, 4, labels=['low', 'avg', 'high', 'very high'])

    # drop one redundant column
    df.drop(columns='Data.Employed.Asian.Unemployment Rate')

    # concat year and month
    df['date'] = df['month'].astype(str).str.cat(df['year'].astype(str), sep=', ')
    
    # set date to datetime index
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    
    # download cleaned data to a .csv
    df.to_csv(filename, index=False)
    
    print('Downloading data from SQL...')
    print('Saving to .csv')
    return df