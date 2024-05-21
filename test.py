# Including the necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Import data, make a copy of the original
df0 = pd.read_csv('seattle-weather.csv')
dfc1 = df0.copy()
dfc1.head()

# Checking the unique values in the 'weather' column
unique_fields = dfc1['weather'].unique()
print(unique_fields)

# Put categorical varaibles in a list
categorical_lst = ['date','weather']
# Create a seperate & smaller dataframe for categorical variables
dfc2a = pd.DataFrame(dfc1, columns=categorical_lst, copy=True)
dfc2a.head()

# Put all continuous variables into a list
continuous_lst = ['precipitation', 'temp_max', 'temp_min', 'wind']
# Create a seperate & smaller dataframe for our chosen variables. Use 'copy=True' so changes wont affect original
dfc2b = pd.DataFrame(dfc1, columns=continuous_lst, copy=True)
dfc2b.head()

# Create new df with variables we want to work with:
new_cols = ['date', 'precipitation', 'temp_max', 'temp_min', 'wind', 'weather']
df = df0[new_cols]
# df.head()

# Let's show all columns with missing data as well:
df[df.isnull().any(axis=1)]
df.isnull().any()
num_stdv = 1

# Define the labels dictionary
labels = {
   'precipitation': ['low', 'mid', 'high'],
   'temp_max': ['low', 'mid', 'high'],
   'temp_min': ['low', 'mid', 'high'],
   'wind': ['low', 'mid', 'high']
}

# Create bounds for continuous labels
# Create bounds for continuous labels
for col in df.columns:
   if col in labels:
       col_mean = df[col].mean()
       col_stdv = df[col].std()
       lower_bound = col_mean - col_stdv * num_stdv
       upper_bound = col_mean + col_stdv * num_stdv
       bins = [-float('inf'), lower_bound, upper_bound, float('inf')]
       df[col] = pd.cut(df[col], bins=bins, labels=labels[col])
df.head()

model = BayesianNetwork([
        ('weather', 'precipitation'),
        ('weather', 'wind'),
        ('precipitation', 'temp_max'),
        ('wind', 'temp_min')
    ])
weather_states = ['drizzle', 'rain', 'fog', 'snow', 'sun']
precipitation_states = ['low', 'mid', 'high']
temp_max_states =['low', 'mid', 'high']
temp_min_states =['low', 'mid', 'high']
wind_states = ['low', 'mid', 'high']


weather_marginal = (df['weather'].value_counts()/len(df['weather'])).round(3)
weather_marginal = np.array([[value] for value in weather_marginal])
print(weather_marginal)
var_dict = {'weather': ['precipitation', 'wind'],
            'precipitation': ['temp_max'],
            'wind': ['temp_min'],}
cpd_lst = []
for key, value in var_dict.items():
    length = len(value)
    for i in range(length):
        value_given_key = df.groupby(key)[value[i]].value_counts(
                                                    normalize=True
                                                    ).sort_index()
        cpd = value_given_key.unstack(fill_value=0).to_numpy().T
        cpd_lst.append(cpd)

cpd_lst[2][:,0] = .33
print(cpd_lst)

# Creating tabular conditional probability distribution
weather_cpd = TabularCPD(variable='weather', variable_card=5, values=weather_marginal, state_names={'weather': weather_states})

precipitation_cpd = TabularCPD(variable='precipitation', variable_card=3, evidence=['weather'], evidence_card=[5], values=cpd_lst[0], state_names={'precipitation': precipitation_states, 'weather': weather_states})

wind_cpd = TabularCPD(variable='wind', variable_card=3, evidence=['weather'], evidence_card=[5], values=cpd_lst[1], state_names={'wind': wind_states, 'weather': weather_states})

temp_max_cpd = TabularCPD(variable='temp_max', variable_card=3, evidence=['precipitation'], evidence_card=[3], values=cpd_lst[2], state_names={'temp_max': temp_max_states, 'precipitation': precipitation_states})

temp_min_cpd = TabularCPD(variable='temp_min', variable_card=3, evidence=['wind'], evidence_card=[3], values=cpd_lst[3], state_names={'temp_min': temp_min_states, 'wind': wind_states})

# cpds = [weather_cpd, wind_cpd, precipitation_cpd, temp_min_cpd, temp_max_cpd]
model.add_cpds(weather_cpd, wind_cpd, precipitation_cpd, temp_min_cpd, temp_max_cpd)
model.check_model()
print(model.nodes())
print(model.edges())
# Print the probability table of the weather node
print(weather_cpd)

# Print the probability table of the wind node
print(wind_cpd)

