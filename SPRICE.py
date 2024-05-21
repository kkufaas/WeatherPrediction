# Including the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.sampling import GibbsSampling
from pgmpy.inference import BeliefPropagation
from pgmpy.sampling import BayesianModelSampling
import logging

logger = logging.getLogger('pgmpy')
logger.setLevel(logging.ERROR)

# Define the labels dictionary
labels = {
    'Air_temp_Act': ['cold', 'mild', 'warm', 'hot'],  # can be important for maritime sector. Independent variable.
    'Rel_Humidity_act': ['low', 'mid', 'high'],
    'Rel_Air_Pressure': ['low', 'mid', 'high'],
    'Wind_Speed_avg': ['calm', 'very low', 'low', 'moderate', 'fresh'],  # can be important for maritime sector
    'Precipitation_Intensity': ['low', 'mid', 'high'],
    'Precipitation_Type': ['no', 'rain', 'snow'],
    'Wind_Direction_vct': ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
}


def import_data():
    df0 = pd.read_csv('SPRICE_Norwegian_Maritime_Data.csv', low_memory=False)
    return df0


def categorize_data():
    df0 = import_data()
    dfc1 = df0.copy()

    bins = [-0.1, 45, 90, 135, 180, 225, 270, 315, 360]
    wind_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

    wind_speed_bins = [0, 0.5, 3, 5, 8, 10]
    wind_speed_labels = ['calm', 'very low', 'low', 'moderate', 'fresh']

    temperature_bins = [-10, 0, 10, 20, 30]
    temperature_labels = ['cold', 'mild', 'warm', 'hot']

    precipitation_type_bins = [-0.1, 59.5, 69.5, np.inf]
    precipitation_type_labels = ['no', 'rain', 'snow']

    # Categorize Wind Direction
    df0['Wind_Direction_vct'] = pd.cut(df0['Wind_Direction_vct'], bins=bins, labels=wind_labels,
                                       include_lowest=True)
    # Categorize Wind Speed
    df0['Wind_Speed_avg'] = pd.cut(df0['Wind_Speed_avg'], bins=wind_speed_bins, labels=wind_speed_labels,
                                   include_lowest=True)
    # Categorize Temperature
    df0['Air_temp_Act'] = pd.cut(df0['Air_temp_Act'], bins=temperature_bins, labels=temperature_labels,
                                 include_lowest=True)
    # Categorize Precipitation type
    df0['Precipitation_Type'] = pd.cut(df0['Precipitation_Type'], bins=precipitation_type_bins,
                                       labels=precipitation_type_labels,
                                       include_lowest=True)

    # Put categorical varaibles in a list
    categorical_lst = ['TIMESTAMP']

    # Create a seperate & smaller dataframe for categorical variables
    dfc2a = pd.DataFrame(dfc1, columns=categorical_lst, copy=True)
    dfc2a.head()

    # Put the rest of continuous variables into a list
    continuous_lst = ['Rel_Humidity_act', 'Rel_Air_Pressure',
                      'Precipitation_Intensity']
    # Create a seperate & smaller dataframe for our chosen variables. Use 'copy=True' so changes wont affect original
    dfc2b = pd.DataFrame(dfc1, columns=continuous_lst, copy=True)
    dfc2b.head()

    # Create new df with variables we want to work with:
    new_cols = ['TIMESTAMP', 'Air_temp_Act', 'Rel_Humidity_act', 'Rel_Air_Pressure', 'Wind_Speed_avg',
                'Precipitation_Type', 'Precipitation_Intensity', 'Wind_Direction_vct']
    df = df0[new_cols]
    print_full(new_cols)

    # Let's show all columns with missing data as well:
    df[df.isnull().any(axis=1)]
    df.isnull().any()
    num_stdv = 1

    # Create bounds for continuous labels
    for col in df.columns:
        if col in labels and col not in ['Wind_Direction_vct', 'Wind_Speed_avg', 'Air_temp_Act', 'Precipitation_Type']:
            col_mean = df[col].mean()
            col_stdv = df[col].std()
            lower_bound = col_mean - col_stdv * num_stdv
            upper_bound = col_mean + col_stdv * num_stdv
            bins = [-float('inf'), lower_bound, upper_bound, float('inf')]
            # df[col] = pd.cut(df[col], bins=bins, labels=labels[col])
            df.loc[:, col] = pd.cut(df[col], bins=bins, labels=labels[col])
    # df.head()
    # print_full(df)
    return df


def set_cpds():
    df = categorize_data()
    model = BayesianNetwork([
        ('Air_temp_Act', 'Rel_Air_Pressure'),
        ('Air_temp_Act', 'Rel_Humidity_act'),
        ('Wind_Speed_avg', 'Rel_Humidity_act'),
        ('Wind_Direction_vct', 'Wind_Speed_avg'),
        ('Rel_Humidity_act', 'Precipitation_Type'),
        ('Precipitation_Type', 'Precipitation_Intensity')
    ])

    air_temp_states = ['cold', 'mild', 'warm', 'hot']
    air_pressure_states = ['low', 'mid', 'high']
    humidity_states = ['low', 'mid', 'high']
    wind_speed_states = ['calm', 'very low', 'low', 'moderate', 'fresh']
    precipitation_intensity_states = ['low', 'mid', 'high']
    precipitation_type_states = ['no', 'rain', 'snow']
    wind_direction_states = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

    var_dict = {
        'Air_temp_Act': ['Rel_Air_Pressure'],
        ('Air_temp_Act', 'Wind_Speed_avg'): ['Rel_Humidity_act'],
        'Wind_Direction_vct': ['Wind_Speed_avg'],
        'Rel_Humidity_act': ['Precipitation_Type'],
        'Precipitation_Type': ['Precipitation_Intensity']
    }

    cpd_lst = []
    for key, values in var_dict.items():
        for value in values:
            if not isinstance(key, tuple):  # Single parent scenario
                value_given_key = df.groupby(key)[value].value_counts(normalize=True).sort_index()
                cpd = value_given_key.unstack(fill_value=0).to_numpy().T

                # Retrieve appropriate labels for the parent and child categories
                # parent_labels = labels.get(key, ['Unknown State'])  # Fallback to 'Unknown State' if not found
                # child_labels = labels.get(value, ['Unknown State'])  # Fallback to 'Unknown State' if not found

                # Transpose only if the parent node is 'Wind_Direction_vct'
                nodes_to_transpose = ['Wind_Direction_vct', 'Wind_Speed_avg', 'Precipitation_Type']
                if key in nodes_to_transpose:
                    cpd = cpd.T  # Transpose to match expected dimensions

                # print(f"CPD for '{value}' given '{key}':")
                # print(pd.DataFrame(cpd, index=parent_labels, columns=child_labels))  # Use actual labels here
                # print("\n")
                cpd_lst.append(cpd)

            else:
                # Multiple parents scenario
                grouped = df.groupby(list(key))[value].value_counts(normalize=True).unstack(fill_value=0)
                cpd = grouped.values.flatten()
                # print(f"CPD for '{value}' given '{', '.join(key)}':")
                # print(pd.DataFrame(grouped).fillna(0))
                # print("\n")
                if key == 'Air_temp_Act':
                    cpd = cpd.T
                cpd_lst.append(cpd)
    # prec_int = cpd_lst[4].T
    # prec_int[:, 2] = .33
    # print(cpd_lst)

    # Marginal probability for 'Air_temp_Act'
    air_temp_counts = df['Air_temp_Act'].value_counts(normalize=True)
    air_temp_marginal = np.array([air_temp_counts.get(state, 0) for state in air_temp_states]).reshape(4, 1)

    # Marginal probability for 'Wind_Direction_vct'
    wind_direction_counts = df['Wind_Direction_vct'].value_counts(normalize=True)
    wind_direction_marginal = np.array(
        [wind_direction_counts.get(state, 0) for state in wind_direction_states]).reshape(8,
                                                                                          1)

    wind_direction_cpd = TabularCPD(variable='Wind_Direction_vct', variable_card=8, values=wind_direction_marginal,
                                    state_names={'Wind_Direction_vct': wind_direction_states})

    air_temp_cpd = TabularCPD(variable='Air_temp_Act', variable_card=4, values=air_temp_marginal,
                              state_names={'Air_temp_Act': air_temp_states})

    # Print independend CPDs to verify
    # print("Wind Direction CPD:\n", wind_direction_cpd)
    print("Air Temperature CPD:\n", air_temp_cpd)

    wind_speed = cpd_lst[2].T
    precipitation_intensity = cpd_lst[4].T

    wind_speed_cpd = TabularCPD(variable='Wind_Speed_avg', variable_card=5, evidence=['Wind_Direction_vct'],
                                evidence_card=[8],
                                values=wind_speed,
                                state_names={'Wind_Speed_avg': wind_speed_states,
                                             'Wind_Direction_vct': wind_direction_states})
    # print("Wind Speed CPD: \n")
    # print_full(wind_speed_cpd)

    air_pressure_cpd = TabularCPD(variable='Rel_Air_Pressure', variable_card=3, evidence=['Air_temp_Act'],
                                  evidence_card=[4],
                                  values=cpd_lst[0],
                                  state_names={'Rel_Air_Pressure': air_pressure_states,
                                               'Air_temp_Act': air_temp_states})
    # print("Air Pressure CPD:\n")
    # print_full(air_pressure_cpd)

    humidity = [cpd_lst[1][i:i + 3] for i in range(0, len(cpd_lst[1]), 3)]
    humidity_transposed = [list(x) for x in zip(*humidity)]
    humidity_cpd = TabularCPD(variable='Rel_Humidity_act', variable_card=3, evidence=['Air_temp_Act', 'Wind_Speed_avg'],
                              evidence_card=[4, 5],
                              values=humidity_transposed,
                              state_names={'Rel_Humidity_act': humidity_states,
                                           'Air_temp_Act': air_temp_states,
                                           'Wind_Speed_avg': wind_speed_states}
                              )
    # print("Humidity CPD: \n")
    # print_full(humidity_cpd)
    precipitation_type_cpd = TabularCPD(variable='Precipitation_Type', variable_card=3, evidence=['Rel_Humidity_act'],
                                        evidence_card=[3],
                                        values=cpd_lst[3],
                                        state_names={'Precipitation_Type': precipitation_type_states,
                                                     'Rel_Humidity_act': humidity_states})

    # print("Precipitation Type CPD:\n")
    # print_full(precipitation_type_cpd)

    precipitation_intensity_cpd = TabularCPD(variable='Precipitation_Intensity', variable_card=3,
                                             evidence=['Precipitation_Type'],
                                             evidence_card=[3],
                                             values=precipitation_intensity,
                                             state_names={'Precipitation_Intensity': precipitation_intensity_states,
                                                          'Precipitation_Type': precipitation_type_states})

    # print("Precipitation Intensity CPD:\n")
    # print_full(precipitation_intensity_cpd)

    model.add_cpds(air_temp_cpd, wind_speed_cpd, wind_direction_cpd, air_pressure_cpd, humidity_cpd,
                   precipitation_type_cpd,
                   precipitation_intensity_cpd)
    return model, air_temp_cpd, wind_speed_cpd, wind_direction_cpd, air_pressure_cpd, humidity_cpd, precipitation_type_cpd, precipitation_intensity_cpd


def print_full(cpd):
    backup = TabularCPD._truncate_strtable
    TabularCPD._truncate_strtable = lambda self, x: x
    print(cpd)
    TabularCPD._truncate_strtable = backup


def calculate_probability():  # To check if data is correct
    df = categorize_data()
    rain_count = df[df['Air_temp_Act'] == 'mild'].shape[0]
    rain_high_wind_count = df[(df['Air_temp_Act'] == 'mild') & (df['Rel_Air_Pressure'] == 'high')].shape[0]
    if rain_count > 0:
        rain_high_wind_probability = rain_high_wind_count / rain_count
    else:
        rain_high_wind_probability = 0.0
    return rain_count, rain_high_wind_count, rain_high_wind_probability


def perform_inference(model):
    inference = VariableElimination(model)
    # (a) Probability of low wind speed when wind direction is South
    prob = inference.query(variables=['Wind_Speed_avg'], evidence={'Wind_Direction_vct': 'S'}, show_progress=False)
    index = model.get_cpds('Wind_Speed_avg').state_names['Wind_Speed_avg'].index('low')
    print("\nProbability of low wind speed when wind direction is South: ")
    probability = prob.values[index]
    print(probability)

    # (b) Probability of South wind direction when the wind speed is low
    prob1 = inference.query(variables=['Wind_Direction_vct'], evidence={'Wind_Speed_avg': 'low'}, show_progress=False)
    index1 = model.get_cpds('Wind_Direction_vct').state_names['Wind_Direction_vct'].index('S')
    print("\nProbability of South wind direction when the wind speed is low: ")
    probability1 = prob1.values[index1]
    print(probability1)
    return probability, probability1


def calculate_joint_probabilities(model):
    # Initialize the Variable Elimination inference
    inference = VariableElimination(model)
    # (a) Calculate the joint probability distribution for all variables of interest
    joint_prob = inference.query(variables=['Wind_Speed_avg', 'Wind_Direction_vct'],
                                 show_progress=False)
    # Extract values and variable assignments
    joint_values = joint_prob.values
    assignments = joint_prob.state_names
    # Create DataFrame from the assignments and values
    index_product = pd.MultiIndex.from_product(
        [assignments[var] for var in ['Wind_Speed_avg', 'Wind_Direction_vct']],
        names=['Wind_Speed_avg', 'Wind_Direction_vct']
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


def temperature_given_high_air_pressure(model):
    inference = VariableElimination(model)
    # Calculate the conditional probability distribution
    prob = inference.query(variables=['Air_temp_Act'], evidence={'Rel_Air_Pressure': 'high'},
                           show_progress=False)
    # Print the conditional probability distribution
    print("Conditional probabilities of air temperature given high air pressure:")
    print(prob)
    return prob


def air_temperature_humidity_and_air_pressure(model):
    inference = VariableElimination(model)
    prob = inference.query(
        variables=['Air_temp_Act'],
        evidence={'Rel_Humidity_act': 'low', 'Rel_Air_Pressure': 'high'},
        show_progress=False
    )
    print("Air temperature given low humidity and high air pressure:")
    print(prob)

    prob1 = inference.query(
        variables=['Air_temp_Act'],
        evidence={'Rel_Humidity_act': 'low', 'Rel_Air_Pressure': 'low'},
        show_progress=False
    )
    print("Air temperature given low humidity and low air pressure:")
    print(prob1)
    return prob, prob1


# Task 1.3, Question 1
def rejection_sampling(model, num_samples=10000):
    sampler = BayesianModelSampling(model)
    try:
        # (a) Probability of low wind speed given wind direction is South
        samples = sampler.rejection_sample(evidence=[('Wind_Direction_vct', 'S')], size=num_samples)
        if samples.empty:
            prob = 0
        else:
            prob = (samples['Wind_Speed_avg'] == 'low').mean()
        print(f"\nProbability of low wind speed given wind direction is South: {prob:.4f}")

        # (b) Probability of South wind direction when the wind speed is low
        samples1 = sampler.rejection_sample(evidence=[('Wind_Speed_avg', 'low')], size=num_samples)
        if samples1.empty:
            prob1 = 0
        else:
            prob1 = (samples1['Wind_Direction_vct'] == 'S').mean()
        print(f"\nProbability of South wind direction given wind speed is low: {prob1:.4f}\n")
    except Exception as e:
        print("Error during sampling:", str(e))
        prob, prob1 = None, None
    return prob, prob1


# Task 1.3, Question 2
def forward_sampling(model, num_samples=10000):
    sampler = BayesianModelSampling(model)
    samples = sampler.forward_sample(size=num_samples)
    joint_distribution = samples.groupby(['Wind_Speed_avg', 'Wind_Direction_vct']).size().div(num_samples)
    print(joint_distribution.sort_values(ascending=False))
    return joint_distribution


# Task 1.3, Question 3
# Conditional probabilities of air temperature given high air pressure:
def likelihood_weighting(model, evidence={'Rel_Air_Pressure': 'high'}, sample_size=1000):
    sampler = BayesianModelSampling(model)
    samples = sampler.forward_sample(size=sample_size)
    weights = (samples['Rel_Air_Pressure'] == evidence['Rel_Air_Pressure']).astype(int)
    filtered_samples = samples[weights > 0]
    if filtered_samples.empty:
        print("No samples match the given evidence. Try increasing the sample size.")
        return None
    weather_counts = filtered_samples['Air_temp_Act'].value_counts(normalize=True)
    print("Conditional probabilities of air temperature given high air pressure:")
    print(weather_counts)
    return weather_counts


def normalize_probabilities(probabilities):
    probability_sum = np.sum(probabilities)
    if np.isclose(probability_sum, 1.0):
        return probabilities
    else:
        return probabilities / probability_sum


def map_types(series, map):
    # Convert the indices in the series to the corresponding types
    return series.rename(index=map)


# Task 1.3, Question 4
# Please note this function works in 50% cases.
# If it shows some kind of Nan error, just run it 1-2 times more.
def gibbs_sampling(model, num_samples=10000):
    temperature_types = {
        0: 'cold',
        1: 'mild',
        2: 'warm',
        3: 'hot'
    }
    sampler = GibbsSampling(model)
    samples = sampler.sample(size=num_samples)
    low_index = 0
    mid_samples = samples[samples['Rel_Humidity_act'] == low_index]
    if mid_samples.empty:
        print("No 'low' humidity samples found.")
        return None, None

    pressure_low = 0
    pressure_high = 2
    weights_low = (mid_samples['Rel_Air_Pressure'] == pressure_low).astype(float)
    weights_high = (mid_samples['Rel_Air_Pressure'] == pressure_high).astype(float)

    weights_low = normalize_probabilities(weights_low)
    weights_high = normalize_probabilities(weights_high)

    resampled_low = mid_samples.sample(n=num_samples, replace=True, weights=weights_low)
    resampled_mid = mid_samples.sample(n=num_samples, replace=True, weights=weights_high)

    probabilities_low_pressure = resampled_low['Air_temp_Act'].value_counts(normalize=True)
    probabilities_high_pressure = resampled_mid['Air_temp_Act'].value_counts(normalize=True)

    probabilities_low_pressure = map_types(probabilities_low_pressure, temperature_types)
    probabilities_high_pressure = map_types(probabilities_high_pressure, temperature_types)

    print("Air temperature given low humidity and low air pressure:")
    print(probabilities_low_pressure)
    print("Air temperature given low humidity and high air pressure:")
    print(probabilities_high_pressure)
    return probabilities_low_pressure, probabilities_high_pressure


if __name__ == "__main__":
    ############################### Draw the model ##################################################
    # # Node positions
    # positions = {
    #     'Air_temp_Act': (1, 4),
    #     'Rel_Air_Pressure': (0, 3),
    #     'Rel_Humidity_act': (2, 3),
    #     'Wind_Speed_avg': (3, 2),
    #     'Wind_Direction_vct': (4, 3),
    #     'Precipitation_Type': (3, 1),
    #     'Precipitation_Intensity': (3, 0)
    # }
    #
    # # Connections between nodes (edges of the Bayesian Network)
    # edges = [
    #     ('Air_temp_Act', 'Rel_Air_Pressure'),
    #     ('Air_temp_Act', 'Rel_Humidity_act'),
    #     ('Wind_Speed_avg', 'Rel_Humidity_act'),
    #     ('Wind_Direction_vct', 'Wind_Speed_avg'),
    #     ('Rel_Humidity_act', 'Precipitation_Type'),
    #     ('Precipitation_Type', 'Precipitation_Intensity')
    # ]
    # # Create figure and axes
    # fig, ax = plt.subplots()
    # # Draw nodes
    # for node, (x, y) in positions.items():
    #     ax.text(x, y, node, ha='center', va='center', fontsize=12,
    #             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='black'))
    # # Draw edges
    # for start, end in edges:
    #     start_pos = positions[start]
    #     end_pos = positions[end]
    #     arrow_style = {"arrowstyle": "-|>", "color": "black", "lw": 1}
    #     ax.annotate('', xy=end_pos, xycoords='data', xytext=start_pos, textcoords='data',
    #                 arrowprops=arrow_style)
    # # Set plot parameters
    # ax.set_xlim(-1, 4)
    # ax.set_ylim(-1, 5)
    # ax.axis('off')  # Hide axes
    # # Display the plot
    # plt.show()
    ################################################################################################
    model, air_temp_cpd, wind_speed_cpd, wind_direction_cpd, air_pressure_cpd, humidity_cpd, precipitation_type_cpd, \
        precipitation_intensity_cpd = set_cpds()
    print("\nModel Check:", model.check_model())
    print("\nNodes and Edges:", model.nodes(), model.edges())

    # print("\nAir Temp CPD: ")
    # print_full(air_temp_cpd)
    #
    # print("\nAir Pressure CPD: ")
    # print_full(air_pressure_cpd)

    # calculate_probability()

    # Task 1.2 question 1
    results = perform_inference(model)

    # Task 1.2 question 2
    joint_probabilities, most_probable = calculate_joint_probabilities(model)
    # Among all the combinations evaluated, mid-level combination is most commonly recorded
    #
    # # Task 1.2 question 3
    conditional_probabilities = temperature_given_high_air_pressure(model)

    # # Task 1.2 question 4
    air_temperature_humidity_and_air_pressure(model)

    # Task 1.3 question 1
    rejection_sampling(model)
    # exact inference:
    # Probability of low wind speed when wind direction is South: 0.16674571648781644
    # Probability of South wind direction when the wind speed is low: 0.2620279849600298
    # approximate inference:
    # Probability of low wind speed given wind direction is South: 0.1716
    # Probability of South wind direction given wind speed is low: 0.2661

    # Task 1.3 question 2
    forward_sampling(model)
    # 0.124892 exact inference vs 0.1305 approximate inference

    # Task 1.3 question 3
    # Conditional probabilities of air temperature given high air pressure:
    try:
        likelihood_weighting(model)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # exact
        # mild 0.6643
        # warm 0.2830
        # cold 0.0526
        # vs approximate Air_temp_Act
        # mild    0.610778
        # warm    0.341317
        # cold    0.047904

    # Task 1.3 question 4
    # Please note this function works in 50% cases.
    # If it shows some kind of Nan error, just run it 1-2 times more.
    try:
        low_pressure, high_pressure = gibbs_sampling(model)
        if low_pressure is not None and high_pressure is not None:
            print("Sampling and resampling completed successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


'''
Exact values:
Air temperature given low humidity and high air pressure:
+--------------------+---------------------+
| Air_temp_Act       |   phi(Air_temp_Act) |
+====================+=====================+
| Air_temp_Act(cold) |              0.1268 |
+--------------------+---------------------+
| Air_temp_Act(mild) |              0.6818 |
+--------------------+---------------------+
| Air_temp_Act(warm) |              0.1914 |
+--------------------+---------------------+
| Air_temp_Act(hot)  |              0.0000 |
+--------------------+---------------------+
Air temperature given low humidity and low air pressure:
+--------------------+---------------------+
| Air_temp_Act       |   phi(Air_temp_Act) |
+====================+=====================+
| Air_temp_Act(cold) |              0.2886 |
+--------------------+---------------------+
| Air_temp_Act(mild) |              0.2170 |
+--------------------+---------------------+
| Air_temp_Act(warm) |              0.2588 |
+--------------------+---------------------+
| Air_temp_Act(hot)  |              0.2355 |
+--------------------+---------------------+

vs 
Approximate inference:

Air temperature given low humidity and high air pressure:
Air_temp_Act
mild    0.7137
warm    0.1941
cold    0.0922


Air temperature given low humidity and low air pressure:
Air_temp_Act
warm    0.3001
cold    0.2586
hot     0.2222
mild    0.2191


'''

###################################### Correlation matrix ##########################################################
# df = import_data()
# correlation_matrix = df[['Air_temp_Act', 'Rel_Humidity_act', 'Rel_Air_Pressure','Wind_Speed_avg', 'Precipitation_Type', 'Precipitation_Intensity', 'Wind_Direction_vct']].corr()
# # Plot the correlation matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
# plt.title('Correlation Matrix of Environmental Variables')
# plt.show()
################################################################################################
