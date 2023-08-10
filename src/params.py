"""Input parameters for regression."""


#filename_ip_data = "../inputfile/automobileEDA.csv"
#X_columns = ['highway-mpg', 'engine-size', 'horsepower', 'curb-weight' ]
#y_column = 'price'

filename_ip_data = "../inputfile/FuelConsumptionCo2.csv"
X_columns = ['ENGINESIZE', 'FUELCONSUMPTION_COMB', 'CYLINDERS' ]
y_column = 'CO2EMISSIONS'
num_run = 10  # Number of run for random_split
test_size = 0.2  # Test size for train_test_split
seed = 0  # Random seed for reproducibility