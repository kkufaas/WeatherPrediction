# Including the necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.sampling import GibbsSampling
from pgmpy.inference import BeliefPropagation
from pgmpy.sampling import BayesianModelSampling
import logging

logger = logging.getLogger('pgmpy')
logger.setLevel(logging.ERROR)


def import_data():
    df0 = pd.read_csv('seattle-weather.csv')
    return df0


def categorize_data():
    # Define the labels dictionary
    labels = {
        'precipitation': ['low', 'mid', 'high'],
        'temp_max': ['low', 'mid', 'high'],
        'temp_min': ['low', 'mid', 'high'],
        'wind': ['low', 'mid', 'high']
    }

    # Put categorical varaibles in a list
    df0 = import_data()
    dfc1 = df0.copy()
    categorical_lst = ['date', 'weather']
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

    # Let's show all columns with missing data as well:
    df[df.isnull().any(axis=1)]
    df.isnull().any()
    num_stdv = 1

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
    return df


def normalize_cpd_data(cpd_data):
    cpd_array = np.array(cpd_data)
    column_sums = cpd_array.sum(axis=0)
    column_sums[column_sums == 0] = 1
    normalized_cpd = cpd_array / column_sums
    return normalized_cpd.tolist()


def normalize_probabilities(probabilities):
    probability_sum = np.sum(probabilities)
    if np.isclose(probability_sum, 1.0):
        return probabilities
    else:
        return probabilities / probability_sum


def set_cpds():
    df = categorize_data()
    # Task 1.4.1 Hierarchy 1
    model = BayesianNetwork([
        ('weather', 'precipitation'),
        ('weather', 'wind'),
        ('precipitation', 'temp_max'),
        ('precipitation', 'temp_min'),
        ('wind', 'temp_max'),
        ('wind', 'temp_min')
    ])

    weather_states = ['drizzle', 'fog', 'rain', 'snow', 'sun']
    precipitation_states = ['low', 'mid', 'high']
    temp_max_states = ['low', 'mid', 'high']
    temp_min_states = ['low', 'mid', 'high']
    wind_states = ['low', 'mid', 'high']

    weather_marginal = (df['weather'].value_counts() / len(df['weather'])).round(3)
    weather_marginal = np.array([[value] for value in weather_marginal])
    # print(weather_marginal)

    var_dict = {
        'weather': ['precipitation', 'wind'],
        ('precipitation', 'wind'): ['temp_max', 'temp_min']
    }

    cpd_lst = []
    for key, values in var_dict.items():
        for value in values:
            if not isinstance(key, tuple):  # Single parent scenario
                value_given_key = df.groupby(key)[value].value_counts(normalize=True).sort_index()
                cpd = value_given_key.unstack(fill_value=0).to_numpy().T
                cpd_lst.append(cpd)
            else:
                # Multiple parents scenario
                grouped = df.groupby(list(key))[value].value_counts(normalize=True).unstack(fill_value=0)
                cpd = grouped.values.flatten()
                cpd_lst.append(cpd)

    cpd_data_for_temp_max = np.array(cpd_lst[2]).reshape((3, 9))
    cpd_data_for_temp_min = np.array(cpd_lst[3]).reshape((3, 9))
    normalized_temp_max_cpd = normalize_cpd_data(cpd_data_for_temp_max)
    normalized_temp_min_cpd = normalize_cpd_data(cpd_data_for_temp_min)

    # Creating tabular conditional probability distribution
    weather_cpd = TabularCPD(variable='weather', variable_card=5, values=weather_marginal,
                             state_names={'weather': weather_states})
    precipitation_cpd = TabularCPD(variable='precipitation', variable_card=3, evidence=['weather'], evidence_card=[5],
                                   values=cpd_lst[0],
                                   state_names={'precipitation': precipitation_states, 'weather': weather_states})
    wind_cpd = TabularCPD(variable='wind', variable_card=3, evidence=['weather'], evidence_card=[5],
                          values=cpd_lst[1],
                          state_names={'wind': wind_states, 'weather': weather_states})

    temp_max_cpd = TabularCPD(
        variable='temp_max', variable_card=3, evidence=['precipitation', 'wind'],
        evidence_card=[3, 3], values=normalized_temp_max_cpd,
        state_names={'temp_max': ['low', 'mid', 'high'], 'precipitation': ['low', 'mid', 'high'],
                     'wind': ['low', 'mid', 'high']}
    )

    temp_min_cpd = TabularCPD(
        variable='temp_min', variable_card=3, evidence=['precipitation', 'wind'],
        evidence_card=[3, 3], values=normalized_temp_min_cpd,
        state_names={'temp_min': ['low', 'mid', 'high'], 'precipitation': ['low', 'mid', 'high'],
                     'wind': ['low', 'mid', 'high']}
    )

    model.add_cpds(weather_cpd, wind_cpd, precipitation_cpd, temp_max_cpd, temp_min_cpd)
    return model, cpd_lst, weather_cpd, wind_cpd, precipitation_cpd, temp_max_cpd, temp_min_cpd


def print_full(cpd):
    backup = TabularCPD._truncate_strtable
    TabularCPD._truncate_strtable = lambda self, x: x
    print(cpd)
    TabularCPD._truncate_strtable = backup


def calculate_probability():
    df = categorize_data()
    rain_count = df[df['weather'] == 'snow'].shape[0]
    rain_high_wind_count = df[(df['weather'] == 'snow') & (df['precipitation'] == 'low')].shape[0]
    if rain_count > 0:
        rain_high_wind_probability = rain_high_wind_count / rain_count
    else:
        rain_high_wind_probability = 0.0
    return rain_count, rain_high_wind_count, rain_high_wind_probability


# Task 1.2, Question 1
def perform_inference(model):
    inference = VariableElimination(model)

    # (a) Probability of high wind when the weather is sunny
    prob_high_wind_given_sunny = inference.query(variables=['wind'], evidence={'weather': 'sun'}, show_progress=False)
    high_wind_index = model.get_cpds('wind').state_names['wind'].index('high')
    print("\nProbability of high wind when weather is sunny: ")
    high_wind_probability = prob_high_wind_given_sunny.values[high_wind_index]
    print(high_wind_probability)

    # (b) Probability of sunny weather when the wind is high
    prob_sunny_given_high_wind = inference.query(variables=['weather'], evidence={'wind': 'high'}, show_progress=False)
    sunny_index = model.get_cpds('weather').state_names['weather'].index('sun')
    print("\nProbability of sunny weather when wind is high: ")
    sunny_probability = prob_sunny_given_high_wind.values[sunny_index]
    print(sunny_probability)
    return high_wind_probability, sunny_probability


# Task 1.2, Question 2
def calculate_joint_probabilities(model):
    # Initialize the Variable Elimination inference
    inference = VariableElimination(model)

    # (a) Calculate the joint probability distribution for all variables of interest
    joint_prob = inference.query(variables=['precipitation', 'wind', 'weather'], show_progress=False)
    # Extract values and variable assignments
    joint_values = joint_prob.values
    assignments = joint_prob.state_names
    # Create DataFrame from the assignments and values
    index_product = pd.MultiIndex.from_product(
        [assignments[var] for var in ['precipitation', 'wind', 'weather']],
        names=['precipitation', 'wind', 'weather']
    )
    joint_df = pd.DataFrame(joint_values.flatten(), index=index_product, columns=['probability'])
    joint_df = joint_df.reset_index()
    joint_df = joint_df.sort_values(by='probability', ascending=False).reset_index(drop=True)

    # (b) Find the most probable condition
    most_probable_condition = joint_df.iloc[0]  # The first row after sorting by probability descending
    print("\nAll Possible Joint Probabilities (sorted):")
    print(joint_df)
    print("\nWhere the most probable condition is:")
    print(most_probable_condition)
    return joint_prob, most_probable_condition


# Task 1.2, Question 3
def weather_given_medium_precipitation(model):
    # Initialize the Variable Elimination inference
    inference = VariableElimination(model)
    # Calculate the conditional probability distribution for weather given that precipitation is medium
    weather_given_mid_precip = inference.query(variables=['weather'], evidence={'precipitation': 'mid'},
                                               show_progress=False)
    # Print the conditional probability distribution
    print("Conditional Probabilities of Weather given Medium Precipitation:")
    print(weather_given_mid_precip)
    return weather_given_mid_precip


# Task 1.2, Question 4
def weather_given_medium_precip_and_wind(model):
    # Initialize the Variable Elimination inference
    inference = VariableElimination(model)

    # Probabilities of weather given medium precipitation and low wind
    weather_given_mid_precip_low_wind = inference.query(
        variables=['weather'],
        evidence={'precipitation': 'mid', 'wind': 'low'},
        show_progress=False
    )
    print("Weather given medium precipitation and low wind:")
    print(weather_given_mid_precip_low_wind)

    # Probabilities of weather given medium precipitation and medium wind
    weather_given_mid_precip_mid_wind = inference.query(
        variables=['weather'],
        evidence={'precipitation': 'mid', 'wind': 'mid'},
        show_progress=False
    )
    print("Weather given medium precipitation and medium wind:")
    print(weather_given_mid_precip_mid_wind)

    return weather_given_mid_precip_low_wind, weather_given_mid_precip_mid_wind


# Task 1.3, Question 1
def rejection_sampling(model, num_samples=10000):
    # Initialize the rejection sampler
    sampler = BayesianModelSampling(model)
    try:
        # (a) Probability of high wind when the weather is sunny
        samples_sunny = sampler.rejection_sample(evidence=[('weather', 'sun')], size=num_samples)
        if samples_sunny.empty:
            high_wind_given_sunny = 0
        else:
            high_wind_given_sunny = (samples_sunny['wind'] == 'high').mean()
        print(f"\nProbability of high wind when weather is sunny: {high_wind_given_sunny:.4f}")
        # (b) Probability of sunny weather when the wind is high
        samples_high_wind = sampler.rejection_sample(evidence=[('wind', 'high')], size=num_samples)
        if samples_high_wind.empty:
            sunny_given_high_wind = 0
        else:
            sunny_given_high_wind = (samples_high_wind['weather'] == 'sun').mean()
        print(f"\nProbability of sunny weather when wind is high: {sunny_given_high_wind:.4f}")
    except Exception as e:
        print("Error during sampling:", str(e))
        high_wind_given_sunny, sunny_given_high_wind = None, None
    return high_wind_given_sunny, sunny_given_high_wind


# Task 1.3, Question 2
def forward_sampling(model, num_samples=10000):
    sampler = BayesianModelSampling(model)
    samples = sampler.forward_sample(size=num_samples)
    joint_distribution = samples.groupby(['precipitation', 'wind', 'weather']).size().div(num_samples)
    print(joint_distribution.sort_values(ascending=False))
    return joint_distribution


# Task 1.3, Question 3
def likelihood_weighting(model, evidence={'precipitation': 'mid'}, sample_size=1000):
    sampler = BayesianModelSampling(model)
    samples = sampler.forward_sample(size=sample_size)
    weights = (samples['precipitation'] == evidence['precipitation']).astype(int)
    filtered_samples = samples[weights > 0]
    if filtered_samples.empty:
        print("No samples match the given evidence. Try increasing the sample size.")
        return None
    weather_counts = filtered_samples['weather'].value_counts(normalize=True)
    print("Conditional probabilities of weather given medium precipitation:")
    print(weather_counts)
    return weather_counts


def map_weather_types(series, weather_map):
    # Convert the indices in the series to the corresponding weather types
    return series.rename(index=weather_map)


# Task 1.3, Question 4
# Please note this function works in 50% cases.
# If it shows some kind of Nan error, just run it 1-2 times more.
def gibbs_sampling(model, num_samples=10000):
    weather_types = {
        0: 'drizzle',
        1: 'fog',
        2: 'rain',
        3: 'snow',
        4: 'sun'
    }
    sampler = GibbsSampling(model)
    samples = sampler.sample(size=num_samples)
    mid_index = 1
    mid_samples = samples[samples['precipitation'] == mid_index]
    if mid_samples.empty:
        print("No 'mid' precipitation samples found.")
        return None, None

    wind_low = 0
    wind_mid = 1
    weights_low = (mid_samples['wind'] == wind_low).astype(float)
    weights_mid = (mid_samples['wind'] == wind_mid).astype(float)

    weights_low = normalize_probabilities(weights_low)
    weights_mid = normalize_probabilities(weights_mid)

    resampled_low = mid_samples.sample(n=num_samples, replace=True, weights=weights_low)
    resampled_mid = mid_samples.sample(n=num_samples, replace=True, weights=weights_mid)

    weather_probabilities_low_wind = resampled_low['weather'].value_counts(normalize=True)
    weather_probabilities_mid_wind = resampled_mid['weather'].value_counts(normalize=True)

    weather_probabilities_low_wind = map_weather_types(weather_probabilities_low_wind, weather_types)
    weather_probabilities_mid_wind = map_weather_types(weather_probabilities_mid_wind, weather_types)

    print("Weather given medium precipitation and low wind:")
    print(weather_probabilities_low_wind)
    print("Weather given medium precipitation and medium wind:")
    print(weather_probabilities_mid_wind)
    return weather_probabilities_low_wind, weather_probabilities_mid_wind


if __name__ == "__main__":
    model, cpd_lst, weather_cpd, wind_cpd, precipitation_cpd, temp_min_cpd, temp_max_cpd = set_cpds()
    # print(cpd_lst)
    print("\nModel Check:", model.check_model())
    print("\nNodes and Edges:", model.nodes(), model.edges())

    print("\nWeather CPD: ")
    print(weather_cpd)

    print("\nWind CPD: ")
    print_full(wind_cpd)

    # Task 1.2 question 1
    results = perform_inference(model)

    # Task 1.2 question 2
    joint_probabilities, most_probable = calculate_joint_probabilities(model)

    # Task 1.2 question 3
    conditional_probabilities = weather_given_medium_precipitation(model)

    # Task 1.2 question 4
    # Calculate conditional probabilities for weather given medium precipitation and specific wind conditions
    conditional_probabilities_low, conditional_probabilities_mid = weather_given_medium_precip_and_wind(model)
    print("Results for low wind:")
    print(conditional_probabilities_low)
    print("Results for medium wind:")
    print(conditional_probabilities_mid)

    # Task 1.3 question 1
    rejection_sampling(model)

    # Task 1.3 question 2
    forward_sampling(model)

    # Task 1.3 question 3
    try:
        likelihood_weighting(model)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # Task 1.3 question 4
    # Please note this function works in 50% cases.
    # If it shows some kind of Nan error, just run it 1-2 times more.
    try:
        low_wind, mid_wind = gibbs_sampling(model)
        if low_wind is not None and mid_wind is not None:
            print("Sampling and resampling completed successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
