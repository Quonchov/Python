# Import functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

""" Extract the data """
airbnb_data = pd.read_csv("C:\\Users\\ochie\\OneDrive\\Desktop\\Python\\machine_learning_zoomcamp\\HW2\\AB_NYC_2019.csv")

# Replace the spaces with the underscores and make them lowercase
airbnb_data.columns = airbnb_data.columns.str.lower().str.replace(" ","_")

# Find the datatypes
airbnb_data.dtypes    # Shows me the datatypes 
strings_check = list(airbnb_data.dtypes[airbnb_data.dtypes == "object"].index)

# Loop over the strings_check to format all the info in the dataset
for col in strings_check:
    airbnb_data[col] = airbnb_data[col].str.lower().str.replace(" ","_")
    
# Look at each column and understand whats going on
# for col in airbnb_data.columns:
    # print(col)
    # print(airbnb_data[col].head())
    # print(airbnb_data[col].unique()[:5])
    # print(airbnb_data[col].nunique())
    # print()
    

# # Look at the distibution of prices
# sns.histplot(airbnb_data.price, bins=50)  # bins defines the number of equal-width bins
# sns.histplot(airbnb_data.price[airbnb_data.price < 1000], bins=50) # For house prices lower than 1000

# Due to the pattern from the plot yielding a long tail distribution
# We proceed with a modification to get rid of the long tail so our model don't get confused.
""" Apply the log distribution"""
price_logs = np.log1p(airbnb_data.price)
# sns.histplot(price_logs, bins=50) # The long tail is gone

# Get the missing values (Nan)
missing_val = airbnb_data.isnull().sum()
# print(missing_val)


""" Setup the validation framework """
l = len(airbnb_data)  # We need to split l into percentages for training the dataset
l_val = int(l * 0.2) # Validation set, rounded up as an integer
l_test = int(l * 0.2) # Testing set 
l_train = int(l * 0.6) # Training dataset 

# Modify the dataset
l_train = l - l_val - l_test

#  Data frame datasets
df_train = airbnb_data.iloc[:l_train]
df_val = airbnb_data.iloc[l_train:l_train + l_val]
df_test = airbnb_data.iloc[l_train + l_val:]

# Shuffle the dataset, no sequential order needed
idx = np.arange(l)
""" To make the random set reproducable ( have same random numbers)"""
np.random.seed(42)
np.random.shuffle(idx) # shuffles the index
""" The updated dataseet is thus: """
df_train = airbnb_data.iloc[idx[:l_train]]
df_val = airbnb_data.iloc[idx[l_train:l_train + l_val]]
df_test = airbnb_data.iloc[idx[l_train + l_val:]]
# print(len(df_val), len(df_test), len(df_train))

# Reset the index  of the dataset
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Log transformation
y_train = np.log1p(df_train.price.values) # values outputs the specific info for a particular row
y_val = np.log1p(df_val.price.values)
y_test = np.log1p(df_test.price.values)

# Remove the price variable because we don't want the target variable to be used for training purposes
del df_train["price"]
del df_val["price"]
del df_test["price"]


# Select certain features for training using them as baseline for the dataset
dfs = airbnb_data.dtypes
# print(dfs)  # View the datatypes I need the ones with values only
# print(df_train.columns)   # Views the headings
features = ["latitude","longitude","minimum_nights","number_of_reviews","reviews_per_month","calculated_host_listings_count","availability_365"]
x_train = df_train[features].values              # creates a new tabular data for the training dataset
x_train = df_train[features].isnull().sum() # confirm the previous step
# x_train = df_train[features].fillna().values    # take care of the missing NaN values 
# x_train = df_train[features].fillna(0).isnull().sum() # confirm the previous step



""" Question One: Number of missing values? """ 
# missing_feat_val = airbnb_data[features].isnull().sum()
# print(missing_feat_val)



""" Question Two: Find the median (50% percentile) for the variable 'minimum_nights' """
med = np.median(airbnb_data["minimum_nights"])



""" Question 3 """
""" Fill in the missing values with 0 and the median, then train the data to see the difference """

# Training the model
def train_linear_regression(X,Y):
    
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])  # stacks vectors together as columns
    XTX = X.T.dot(X)
    inv_XTX = np.linalg.inv(XTX)
    # Identity_X = XTX.dot(inv_XTX).round(1) # check for the identity matrix, if it is, then the inverse term is correct
    w_full = inv_XTX.dot(X.T).dot(Y) # bias term; this gives us the baseline... contains all the weight
    # print(w_full)
    # print(len(w_full))
    return w_full[0], w_full[1:]

#  Replace the missing values with 0
x_train_zero = df_train[features].fillna(0).values    # take care of the missing NaN values
w0_zero, w_zero = train_linear_regression(x_train_zero,y_train) # With filled zero value
y_pred_zero = w0_zero + x_train_zero.dot(w_zero)    # Prediction


#  Replace the missing values with the median
x_train_med = df_train[features].fillna(med).values    # take care of the missing NaN values
w0_med, w_med = train_linear_regression(x_train_med,y_train) # With filled zero value
y_pred_med = w0_med + x_train_med.dot(w_med)    # Prediction

# plot the prediction and compare the original plot at the top of this not
# sns.histplot(y_pred_zero, label="prediction", color="yellow", alpha=0.5, bins= 50)   # alpha controls the transparency
# sns.histplot(y_pred_med, label="Target", color="blue", alpha=0.5, bins=50)

# Calculate the Mean
# Root Mean Squared Error: Evaluation of the model (for measuring the quality of the model)
def rmse(y,y_pred):
     error = y - y_pred
     se = error ** 2
     mse = se.mean()
     return np.sqrt(mse)

RMSE_zero = rmse(y_train, y_pred_zero)
# print(f"The mean value after the using first trained dataset to train the data using zero value: {round(RMSE_zero,2)}")

RMSE_med = rmse(y_train, y_pred_med)
# print(f"The mean value after the using first trained dataset to train the data using zero value: {round(RMSE_med,2)}")



# Validating the Model
""" Code for preparing the model: """

# Here we prepare the model before training for zeros value
def prepare_x(df):
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return(X)

x_train_zero1 = prepare_x(df_train)    # preparing dataset
w0_zero1, w_zero1 = train_linear_regression(x_train_zero1, y_train)   # train the model again

x_val_zero1 = prepare_x(df_val)    # prepare the model with the validation dataset
y_pred_zero1 = w0_zero1 + x_val_zero1.dot(w_zero1)    # predict the validation data

RMSE_zero1 = rmse(y_val, y_pred_zero1)
# print(f"The mean value after the using validation dataset to train the data using zero value: {round(RMSE_zero1,2)}")

# Here we prepare the model before training for median value
def prepare_x(df):
    df_num = df[features]
    df_num = df_num.fillna(med)
    X = df_num.values
    return(X)

x_train_med1 = prepare_x(df_train)    # training dataset
w0_med1, w_med1 = train_linear_regression(x_train_med1, y_train)   # train the model again

x_val_med1 = prepare_x(df_val)    # prepare the model with the validation dataset
y_pred_med1 = w0_med1 + x_val_med1.dot(w_med1)    # predict the validation data

RMSE_med1 = rmse(y_val, y_pred_med1)
# print(f"The mean value after the using validation dataset to train the data using median value: {round(RMSE_med1,2)}")

""" Answer: Both are equally good """


""" Question 4 """
""" Train using the Regularized Linear Regression """
# Since the RMSE are equal, use the value of zero for further training
x_train = df_train[features].fillna(0).values    # take care of the missing NaN values with zero values

# Train using regularization
def train_linear_regression_reg(X, y, r=0.001):  # Train for regularization
    
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])  # stacks vectors together as columns
    
    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])
    inv_XTX = np.linalg.inv(XTX)
    # Identity_X = XTX.dot(inv_XTX).round(1) # check for the identity matrix, if it is, then the inverse term is correct
    w_full = inv_XTX.dot(X.T).dot(y) # bias term; this gives us the baseline... contains all the weight
    # print(w_full)
    # print(len(w_full))
    return w_full[0], w_full[1:]

# x_train = prepare_x(df_train)    # training dataset
# w0, w = train_linear_regression_reg(x_train, y_train, r=0.01)   # train the model again, r = 0.01 can be used to change our manipulate our required value, if r is too high an effect occurs
# # The best value of r is required, check the next steps after regularization ===> Tuning the model
# x_val = prepare_x(df_val)    # prepare the model with the validation dataset
# y_pred = w0 + x_val.dot(w)    # predict the validation data

# RMSE = rmse(y_val, y_pred)
# print(f"Updated mean value using regularization: {RMSE}")  # This regularizeed value is better than the previous ones

# The mean value is still high, alter the regularization value, use range of values to alter...
# Try r [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]

# Tuning the Model: Best value for r parameter for the regularization
for r in [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]:
    x_train = prepare_x(df_train)    # training dataset
    w0, w = train_linear_regression_reg(x_train, y_train, r=r)   
    x_val = prepare_x(df_val)    
    y_pred = w0 + x_val.dot(w)   
    RMSE_reg = rmse(y_val, y_pred)
    # print(r, w0, f" Update RMSE: {round(RMSE_reg,2)}")  # w0 is the bias term

# Training the model on the validation set
# select the smallest value for r
r = 0.000001 # From the tuning model above r = 0.000001 seems like a better fit
x_train = prepare_x(df_train)    # training dataset
w0, w = train_linear_regression_reg(x_train, y_train, r=r)   
x_val = prepare_x(df_val)    
y_pred = w0 + x_val.dot(w)   
RMSE_r = rmse(y_val, y_pred)
# print(F"Using r = {r} for the mean, results to {round(RMSE_r,2)}")
# print()

""" Answer: The best value for r is 0.000001 (selected the smallest value for r) """


""" Question 5 """
""" Computing the standard Deviation Note: Standard deviation shows how different the values are. 
    If it's low, then all values are approximately the same. If it's high, the values are different. 
    If standard deviation of scores is low, then our model is stable. """

# Use this to verify the step below
# for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
#     # Shuffle the dataset, no sequential order needed
#     idx = np.arange(l)
#     np.random.seed(seed)
#     np.random.shuffle(idx) # shuffles the index
#     """ The updated dataset is thus: """
#     df_train = airbnb_data.iloc[idx[:l_train]]
#     df_val = airbnb_data.iloc[idx[l_train:l_train + l_val]]
#     df_test = airbnb_data.iloc[idx[l_train + l_val:]]
#     print(f"For seed {seed} \n Trained dataset: {df_train} \n Validation dataset: {df_val} \n Testing dataset: {df_test}")
#     print()

# def seed_val(x):
std = []
for seed_val in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    # Shuffle the dataset, no sequential order needed
    idx = np.arange(l)
    np.random.seed(seed_val)
    np.random.shuffle(idx) # shuffles the index
    
    """ The updated dataset is thus: """
    df_train = airbnb_data.iloc[idx[:l_train]]
    df_val = airbnb_data.iloc[idx[l_train:l_train + l_val]]
    df_test = airbnb_data.iloc[idx[l_train + l_val:]]
    
    # Reset the index  of the dataset
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    # Log transformation
    y_train = np.log1p(df_train.price.values) # values outputs the specific info for a particular row
    y_val = np.log1p(df_val.price.values)
    y_test = np.log1p(df_test.price.values)
    
    # Remove the price variable because we don't want the target variable to be used for training purposes
    del df_train["price"]
    del df_val["price"]
    del df_test["price"]
    
    # Select certain features for training using them as baseline for the dataset
    dfs = airbnb_data.dtypes
    features = ["latitude","longitude","minimum_nights","number_of_reviews","reviews_per_month","calculated_host_listings_count","availability_365"]
    x_train = df_train[features].values              # creates a new tabular data for the training dataset
    x_train = df_train[features].isnull().sum() # confirm the previous step
    
    #  Replace the missing values with 0
    x_train = df_train[features].fillna(0).values    # take care of the missing NaN values
    w0, w = train_linear_regression(x_train,y_train) # With filled zero value
    y_pred_train = w0 + x_train.dot(w)    # Prediction
    
    # plot the prediction and compare the original plot at the top of this not
    # sns.histplot(y_pred_train, label="prediction", color="yellow", alpha=0.5, bins= 50)   # alpha controls the transparency
    
    x_train = prepare_x(df_train)    # prepare dataset
    w0, w = train_linear_regression(x_train, y_train)   # train the model again

    x_val = prepare_x(df_val)    # prepare the model with the validation dataset
    y_pred_val = w0 + x_val.dot(w)    # predict the validation data

    RMSE = rmse(y_val, y_pred_val)
    std.append(RMSE)
    print(f"For the seed {seed_val}: Mean is: {round(RMSE,4)}")


# Calculates the standard deviation 
std = np.std(std)
print(f"The standard deviation is thus: {round(std,3)}")

""" The answer is 0.008 standard deviation, since the value is low that means the values are approximately the same which means our model is stable"""


""" Question Six """
# Using the model
# Here the final model is trained for use, the Train and Validation dataset is used for this process
df_full_train = pd.concat([df_train, df_val]) # Joins the data set from training data and the validation data

# printing the above gives us the result with an index from the validation dataset use reset_index to solve the problem
df_full_train = df_full_train.reset_index(drop=True)
# print(df_full_train)

x_full_train = prepare_x(df_full_train)
# print(x_full_train)

# Train the y_train and y_validation
y_full_train = np.concatenate([y_train, y_val])

# Train the model again
w0_new, w_new = train_linear_regression_reg(x_full_train, y_full_train, r=0.001) 
# print(f"The bias weight term is: {w0}")
# print(f"The weights are: {w}")

# Prepare the training dataset
x_test = prepare_x(df_test)    # training dataset  
y_pred = w0_new + x_test.dot(w_new) 
score = rmse(y_test, y_pred)
print(f"The new rmse on the test dataset is: {round(score,2)}") # This model is better than the other ones so this will be the final model and will be used to predict the price of the car


