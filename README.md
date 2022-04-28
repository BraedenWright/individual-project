# Project Goal
------------
Analyze the U.S. Department of Labor statistics data for the U.S. labor force. Looking at the data, I want to try and predict future employment/ unemployment rate based on previous years data, if possible with the data available. Barring that, analyzing the data may also give insight into what populations have been affected the most when economic With the data being sorted within a timeframe, it will be an excellent opportunity to continue honing my time series analysis skills.

If any other notable trends appear they will certainly be investigated, and suggestions on how to better the dataframe or what other features would be helpful for analysis.



Initial Questions
-----------------
* Is there a noticable trend or seasonality present in the data?

* Has unemployment rate increased for one population over another?

* What gender improved their employment rate the most?

* What demographic (race, gender) has decreased the most in employment rate?

* Do we see an even distribution of employment between different dempographic over time?

# Project Planning
Plan -> Acquire -> Prepare -> Explore -> Deliver
Planning:

***Create a README file***
* Ensure my dataprep modules are well documents and functional

***Acquisition***

* Obtain labor.csv from US Department of Labor database via https://corgis-edu.github.io/corgis/csv/labor/

***Preparation***

* Clean labor data by renaming columns
* Feature Engineer additional columns
* Set datetime index


***Exploration***

* Ask and answer questions about the labor data.
* Visually represent findings with charts. 

***Modeling***

* Build a ML Model to predict possible future trends

***Deliver***

* Deliver a final report with the results of the analysis 


***

# Data Dictionary
--------------

Feature -----> Discription

date -----> Feature engineered datetime index for dataframe 
month_num -----> Numerical representation of Month
month -----> String representation of Month
year -----> Year data was represents
asian_american_cnp -----> Civilian Noninstitutional Population.Asian American
african_american_cnp -----> Civilian Noninstitutional Population.African American
white_cnp -----> Civilian Noninstitutional Population.White
asian_american_not_in_labor_force -----> Count of Asian American population that is not engaged in the civilian labor force
african_american_not_in_labor_force -----> Count of African American  population that is not engaged in the civilian labor force
white_not_in_labor_force -----> Count of white population that is not engaged in the civilian labor force
asian_american_civilian_labor_force -----> Count of Asian American population that IS engaged in the civilian labor force
asian_american_clf_rate -----> Rate of Asian American population that IS engaged in the civilian labor force
asian_american_employed -----> Count of Asian American population that is employed at the time
asian_american_unemployed -----> Count of Asian American population that is unemployed at the time
asian_american_unemployment_rate -----> Unemployment Rate of the Asian American population
all_african_american_civilian_labor_force -----> Count of African American population that IS engaged in the civilian labor force
male_african_american_civilian_labor_force -----> Count of African American, male population that IS engaged in the civilian labor force
female_african_american_civilian_labor_force -----> Count of African American, female population that IS engaged in the civilian labor force
all_african_american_clf_rate -----> Rate of African American population that IS engaged in the civilian labor force
male_african_american_clf_rate -----> Rate of African American, male population that IS engaged in the civilian labor force
female_african_american_clf_rate -----> Rate of African American, female population that IS engaged in the civilian labor force
all_white_civilian_labor_force ----->Count of white population that IS engaged in the civilian labor force
male_white_civilian_labor_force ----->Count of white, male population that IS engaged in the civilian labor force
female_white_civilian_labor_force ----->Count of white, female population that IS engaged in the civilian labor force
all_white_clf_rate -----> Rate of white population that IS engaged in the civilian labor force
male_white_clf_rate -----> Rate of white, male population that IS engaged in the civilian labor force
female_white_clf_rate -----> Rate of white, female population that IS engaged in the civilian labor force
all_african_american_employed ----->Count of African American population that is employed at the time
male_african_american_employed -----> Count of African American, male population that is employed at the time
female_african_american_employed -----> Count of African American, female population that is employed at the time
all_african_american_employment_ratio -----> Employment Rate of the African American population
male_african_american_employment_ratio -----> Employment Rate of the African American, male population
female_african_american_employment_ratio -----> Employment Rate of the African American, female population
all_white_employed -----> Employment Rate of the White population
male_white_employed -----> Employment Rate of the White, Male population
female_white_employed -----> Employment Rate of the White, Female population 
employed_all_white_employment_ratio -----> Employment Rate of the White population
employed_male_white_employment_ratio -----> Employment Rate of the White, Male population
employed_female_white_employment_ratio -----> Employment Rate of the White, Female population
all_african_american_unemployed -----> Count of African American population that is unemployed at the time
male_african_american_unemployed -----> Count of African American, Male population that is unemployed at the time
female_african_american_unemployed -----> Count of African American, Female population that is unemployed at the time
all_african_american_unemployment_rate -----> Unemployment Rate of the African American population
male_african_american_unemployment_rate -----> Unemployment Rate of the African American, Male population
female_african_american_unemployment_rate -----> Unemployment Rate of the African American, Female population
all_white_unemployed -----> Count of White population that is unemployed at the time
male_white_unemployed -----> Count of White, Male population that is unemployed at the time
female_white_unemployed -----> Count of White, Female population that is unemployed at the time 
all_white_unemployment_rate -----> Unemployment Rate of the White population
male_white_unemployment_rate -----> Unemployment Rate of the White, Male population
female_white_unemployment_rate -----> Unemployment Rate of the White, Female population
total_civ_non_population -----> [Feature Engineered]  ['asian_american_cnp'] + ['african_american_cnp'] + ['white_cnp']
total_not_in_labor_force -----> [Feature Engineered]  ['asian_american_not_in_labor_force'] + ['african_american_not_in_labor_force'] + ['white_not_in_labor_force']
avg_clf_rate ----->  [Feature Engineered]  ['asian_american_clf_rate'] + ['all_african_american_clf_rate'] + ['all_white_clf_rate']
total_unemployed_pop -----> [Feature Engineered]  ['asian_american_unemployed'] + ['all_african_american_unemployed'] + ['all_white_unemployed']
avg_female_clf_rate -----> [Feature Engineered]  ['female_african_american_clf_rate'] + ['female_white_clf_rate']
avg_male_clf_rate -----> [Feature Engineered]  ['male_african_american_clf_rate'] + ['male_white_clf_rate']
avg_female_unemployment_rate -----> [Feature Engineered]  ['female_african_american_unemployment_rate'] + ['female_white_unemployment_rate']
avg_male_unemployment_rate -----> [Feature Engineered]  ['male_african_american_unemployment_rate'] + ['male_white_unemployment_rate']
avg_total_unemployment_rate -----> [Feature Engineered]  ['asian_american_unemployment_rate'] + ['all_african_american_unemployment_rate'] + ['all_white_unemployment_rate']
unemployment_bin -----> [Feature Engineered] Categorizes Avg Unemployment Rate by 'low', 'avg', 'high', or 'very high'



# Steps to Reproduce
A copy of the labor.csv first needs to be downloaded from https://corgis-edu.github.io/corgis/csv/labor/

Read this README.md

Download at the wrangle.py and Final_Labor_Report.ipynb file into your working directory

Create a .gitignore to hide undesireables if needed.

Run the Final_Labor_Report.ipynb notebook

***

# Key Findings
* Signs of Seasonality for Unemployment roughly every decade
* African American Unemployment Rates are generally double that of other populations, regardless of gender. This same population will also decrease their Unemployment at a much faster pace
* 
* 


# Recommendation
* The trends and potential seasonality can be used to predict future economic downturns and spikes in the Unemployment Rate.
* We can get so much more from this data if we gather economic information furing the same span of time, especially if you look into what policies were put in place that may have directly lead to a drop in unemployment. 
* We need an accurate reference for Asian American Unemployment Rates, as well as other populations not represented such as LatinX Unemployment Rates


# Next Steps 
* As recommended, find a monthly representation of economic trends for the US and add/compare them to the Labor dataframe. I can even imagine that it's possible to categorize causes of economic downturn, looking for potential variables that might explain the apparent seasonality
* Determine what policies were put in place that may have a direct correlation to the rise and fall of Unemploymnet Rates
* 

