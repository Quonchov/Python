# Class 2 for DataTalk.Club Zoomcamp Machine Learning

import pandas as pd
import numpy as np

#  create a visualization process graphically
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline  # Ensures the plots are displayed in a notebook

dataset = pd.read_csv("C:\\Users\\ochie\\OneDrive\\Desktop\\Python\\machine_learning_zoomcamp\\HW1\\data.csv")


"""First do a data cleaning process make some values lowercases and remove the spaces"""
# Replace the spaces with underscores with a lowercase making the columns unique
dataset.columns = dataset.columns.str.lower().str.replace(" ","_") # Makes the columns lowercase and replaces the space with an underscore
# print(dataset)

# Find the data types
dataset.dtypes  # gives me data types we are interested in the object
strings = list(dataset.dtypes[dataset.dtypes == "object"].index)

# loop of over the strings to format all the infomation in the dataset
for col in strings:
    dataset[col] = dataset[col].str.lower().str.replace(" ","_")

#  Look at each column and understand whats going on
# for col in dataset.columns:
    # print(col)
    # print(dataset[col].head())
    # print(dataset[col].unique()[:5])
    # print(dataset[col].nunique())
    # print()


# Look at the distibution of prices
# sns.histplot(dataset.msrp, bins=50)  # bins defines the number of equal-width bins
# sns.histplot(dataset.msrp[dataset.msrp < 100000], bins=50) # For car prices lower than 100000
# sns.histplot(dataset.msrp[dataset.msrp < 100000], bins=50)

# Due to the pattern from the plot yielding a long tail distribution
# We proceed with a modification to get rid of the long tail so our model dont get confused.
""" Apply the log distribution"""
np.log1p([0, 1, 10, 1000, 100000]) # log1p adds 1 to the values to avoid errors from log(0)
#  the above code is same as np.log([0+1,1+1,10+1,1000+1,100000+1])

price_logs = np.log1p(dataset.msrp)
sns.histplot(price_logs, bins=50)  # the long tail is gone with a consentration of the normal car prices giving a normal distribution which is ideal for model development.

# Getting the missing values (Nan)
missingVal = dataset.isnull().sum()  # Outputs the number of missing values in a dataset
# print(missingVal)  # Missing values are important to consider when training the model





# Setting up the validation framework
l = len(dataset)
""" We will split the training model into three parts: Training set, Validation set and Testing set,
    the training set will be 60% of the dataset, validation set will be 20% and testing set will be 20%
    of the dataset """

l_val = int(l * 0.2) # Validation set, rounded up as an integer
l_test = int(l * 0.2) # Testing set 
l_train = int(l * 0.6) # Training dataset 
# print(l,l_val + l_test + l_train)  # The two won't be the same due to the rounding up
""" modify the dataset using this method """
l_train = l - l_val - l_test # Updated Training dataset 
# print(l_val, l_test, l_train)

#  Data frame datasets
df_train = dataset.iloc[:l_train]
df_val = dataset.iloc[l_train:l_train + l_val]
df_test = dataset.iloc[l_train + l_val:]

#  The dataset are in sequential order, therefore the data needs to be shuffled to include all car models and data
idx = np.arange(l)  # creates an index for the entire dataset
""" To make the random set reproducable ( have same random numbers)"""
np.random.seed(2)
np.random.shuffle(idx) # shuffles the index
""" The updated dataseet is thus: """
df_train = dataset.iloc[idx[:l_train]]
df_val = dataset.iloc[idx[l_train:l_train + l_val]]
df_test = dataset.iloc[idx[l_train + l_val:]]
# print(len(df_val), len(df_test), len(df_train))

#  reset the index  of the dataset
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

#  log transformation
y_train = np.log1p(df_train.msrp.values) # values outputs the specific info for a particular row
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)

# Remove the msrp variable because we don't want the target variable to be used for training purposes
del df_train["msrp"]
del df_val["msrp"]
del df_test["msrp"]





# Linear Regression use th trained datset no need to use the validation or testing dataset
""" Regression used for predicting numbers """
df_train.iloc[10]   # pick some specific values to be used as a feature matrix X

""" Here we want a price of 10000 (prediction) to be returned anytime we put in the values of xi: 
    This will be the linear regression
    
Linear Regression formular: g(xi) = W0 + W1xi1 + W2xi2 + W3xi3 
    thu; g(xi) = W0 + sum(of the three elements)
    W0 is the bias weight and W1, W2, W3 are the sum of the weight bias 
                          

Example of what is expected: """

xi = [435, 11 , 86]
W0 = 1.73
W = [0.01, 0.04, 0.002] # The values given within are the weights W1, W2, W3

# def linear_regression(xi):
#     n = len(xi)
#     pred = W0
    
#     for j in range(n):
#         pred = pred + W[j] * xi[j]
#     # do something
#     return pred

# val = linear_regression(xi) # result of val is still a log value, use the exp to return its original value
# val = np.expm1(val) # Returns the original value


# Linear Regression Vector Form
def dot(xi, W):
    n = len(xi)
    res = 0.0
    
    for j in range(n):
        res = res + xi[j] * W[j]
    return res

# def linear_regression(xi):
#     return W0 + dot(xi, W)


# w_new = [W0] + W

# def linear_regression(xi):
#     xi = [1] + xi
#     return dot(xi, w_new)

# valz = linear_regression(xi)

#  For example
W0 = 7.17
W = [0.01, 0.04, 0.002]
w_new = [W0] + W

x1 = [1, 148, 24, 1385]
x2 = [1, 132, 25, 2031]
x3 = [1, 453, 11, 86]

X = [x1, x2, x3]
X = np.array(X)

def linear_regression(X):
    return X.dot(w_new)

val = linear_regression(X)


#  Training the Linear Regression Model: Check the elements of statistical learning
# We want XW  = y
y = [100, 200, 150, 250, 300, 245, 130, 135, 300]
X = [
 [148, 24, 1385],
 [ 132, 25, 2031],
 [ 453, 11, 86],
 [123, 40, 3485],
 [ 142, 95, 2331],
 [ 373, 91, 84],
 [190, 29, 135],
 [ 232, 22, 231],
 [ 803, 111, 186]
]

X = np.array(X)

def train_linear_regression(X,y):
    
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])  # stacks vectors together as columns
    XTX = X.T.dot(X)
    inv_XTX = np.linalg.inv(XTX)
    # Identity_X = XTX.dot(inv_XTX).round(1) # check for the identity matrix, if it is, then the inverse term is correct
    w_full = inv_XTX.dot(X.T).dot(y) # bias term; this gives us the baseline... contains all the weight
    # print(w_full)
    # print(len(w_full))
    return w_full[0], w_full[1:]

train_linear_regression(X, y)    
# w0 = w_full[0]
# w = w_full[1:] # if the weight is a negative value that means there is a decrease in the total cost 
    
# Using the baseline model for car price prediction
dfs = dataset.dtypes  # select the datatypes with values
df_train.columns  # to view the columns
base = ["engine_hp","engine_cylinders","highway_mpg","city_mpg","popularity"]
x_train = df_train[base].values              # creates a new tabular data for the training dataset
x_train = df_train[base].fillna(0).values    # take care of the missing NaN values 
# x_train = df_train[base].fillna(0).isnull().sum() # confirm the previous step

# Training the model
w0, w = train_linear_regression(x_train, y_train)   # creates the weight bias w0 and sum of the bias w
y_pred = w0 + x_train.dot(w)  # predictions

# plot the prediction and compare the original plot at the top of this not
# sns.histplot(y_pred, label="prediction", color="red", alpha=0.5, bins= 50)   # alpha controls the transparency
# sns.histplot(y_train, label="Target", color="blue", alpha=0.5, bins=50)



# Root Mean Squared Error: Evaluation of the model (for measuring the quality of the model)
def rmse(y,y_pred):
     error = y - y_pred
     se = error ** 2
     mse = se.mean()
     return np.sqrt(mse)

RMSE = rmse(y_train, y_pred)
print(RMSE)

# Validating the model
""" Code for training the model: """

base = ["engine_hp","engine_cylinders","highway_mpg","city_mpg","popularity"]
x_train = df_train[base].values              # creates a new tabular data for the training dataset
x_train = df_train[base].fillna(0).values    # take care of the missing NaN values 

# Training the model
w0, w = train_linear_regression(x_train, y_train)   # creates the weight bias w0 and sum of the bias w
y_pred = w0 + x_train.dot(w)  # predictions

# Here we prepare the model before training 
def prepare_x(df):
    df_num = df[base]
    df_num = df_num.fillna(0)
    X = df_num.values
    return(X)


x_train = prepare_x(df_train)    # training dataset
w0, w = train_linear_regression(x_train, y_train)   # train the model agaim

x_val = prepare_x(df_val)    # prepare the model with the validation dataset
y_pred = w0 + x_val.dot(w)    # predict the validation data

RMSE1 = rmse(y_val, y_pred)
print(RMSE1)


""" "Adding more features to the model: here we will use the year of the car"""
df_train.year.max()    #checking for the year of data release to enable compute the car make year
2017 - df_train.year

def prepare_x(df):
    df = df.copy()
    df["age"] = 2017 - df.year
    features = base + ["age"]
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return(X)

x_train = prepare_x(df_train)  # here the features are modified are passing through the prepare_x function to give 6 features with age inclusive
# print(x_train)
# del df_train["age"]
# print(df_train.columns)

x_train = prepare_x(df_train)    # training dataset
w0, w = train_linear_regression(x_train, y_train)   # train the model agaim

x_val = prepare_x(df_val)    # prepare the model with the validation dataset
y_pred = w0 + x_val.dot(w)    # predict the validation data

RMSE2 = rmse(y_val, y_pred)
print(RMSE2)  # from here the model is improved due to the change in values of the rmse; this can be noted from the plot below
# sns.histplot(y_pred, color="red", alpha=0.5, bins= 50)   # alpha controls the transparency
# sns.histplot(y_val, color="blue", alpha=0.5, bins=50)




""" Categorical Variables """
#  These are the string variable that are in the data and have the object datatypes... number of doors fall among this category
for v in [2, 3, 4]:
    df_train["num_door_%s" % v] = (df_train.number_of_doors == v).astype("int")

def prepare_x(df):
    df = df.copy()
    features = base.copy()
    
    df["age"] = 2017 - df.year
    features.append("age")
    
    for v in [2, 3, 4]:
        df["num_door_%s" % v] = (df.number_of_doors == v).astype("int")
        features.append("num_door_%s" % v)
    
    
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    
    return(X)

prepare_x(df_train)

# train the dataset again
x_train = prepare_x(df_train)    # training dataset
w0, w = train_linear_regression(x_train, y_train)   # train the model agaim

x_val = prepare_x(df_val)    # prepare the model with the validation dataset
y_pred = w0 + x_val.dot(w)    # predict the validation data

RMSE3 = rmse(y_val, y_pred)
print(RMSE3)  # the improvement is not that much, so the number of doors does not really matter


car_make = list(dataset.make.value_counts().head().index)  # the most popular brands of cars
# print(car_make)

# Using the categories make to improve the model
def prepare_x(df):
    df = df.copy()
    features = base.copy()
    
    df["age"] = 2017 - df.year
    features.append("age")
    
    for v in [2, 3, 4]:
        df["num_door_%s" % v] = (df.number_of_doors == v).astype("int")
        features.append("num_door_%s" % v)
        
    for v in car_make:
        df["make_%s" % v] = (df.make == v).astype("int")
        features.append("make_%s" % v)
    
    
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    
    return(X)

prepare_x(df_train)

# train the dataset again
x_train = prepare_x(df_train)    # training dataset
w0, w = train_linear_regression(x_train, y_train)   # train the model again

x_val = prepare_x(df_val)    # prepare the model with the validation dataset
y_pred = w0 + x_val.dot(w)    # predict the validation data

RMSE4 = rmse(y_val, y_pred)
print(RMSE4)  # the improvement is not that much, so the number of doors does not really matter


""" Further Optimization """
df_train.dtypes

category = ["make","engine_fuel_type","transmission_type","driven_wheels","market_category","vehicle_size","vehicle_style"]

categories = {}

for c in category:
    categories[c] = list(dataset[c].value_counts().head().index)   # the list(function) picks the top information 


# Using the car make to improve the model
def prepare_x(df):
    df = df.copy()
    features = base.copy()
    
    df["age"] = 2017 - df.year
    features.append("age")
    
    for v in [2, 3, 4]:
        df["num_door_%s" % v] = (df.number_of_doors == v).astype("int")
        features.append("num_door_%s" % v)
    
    for c, values in categories.items():  
      for v in values:
          df["%s_%s" % (c, v)] = (df[c] == v).astype("int")
          features.append("%s_%s" % (c, v))
    
    
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    
    return(X)

prepare_x(df_train)

# train the dataset again
x_train = prepare_x(df_train)    # training dataset
w0, w = train_linear_regression(x_train, y_train)   # train the model again

x_val = prepare_x(df_val)    # prepare the model with the validation dataset
y_pred = w0 + x_val.dot(w)    # predict the validation data

RMSE5 = rmse(y_val, y_pred)
print(f"This value did not meet up the expectation: {RMSE5}")  # the value here is bad and cannot be implemented, the bias is not in other, fixing is done in the next step


# Regularization: controlling the weights, so they don't grow too much.
"Example: "
X = [
     [2, 1, 1],
     [4, 3, 3],
     [5, 7, 7]
     ]
y = [1,2,3,4,1,2,3]
X = np.array(X)
XTX = X.T.dot(X)   # computes the dot product of X with its transpose X.T
# inv_XTX = np.linalg.inv(XTX) # this will give a singular error due to same vector array of X which leads to large number in the bias weights
# inv_XTX.dot(X.T).dot(y)
" To solve the issue of singular error, include an identity matrix or add a small value to the diagonal of matrix X "
I = np.eye(3)
XTX = XTX + 0.001 * I # improving the weight of the bias- Regularization 0.01 can be used to determine how much regularization is needed
inv_XTX = np.linalg.inv(XTX)
# inv_XTX.dot(X.T).dot(y)

# print(inv_XTX)

def train_linear_regression_reg(X, y, r=0.001):             # Train for regularization
    
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


x_train = prepare_x(df_train)    # training dataset
w0, w = train_linear_regression_reg(x_train, y_train, r=0.01)   # train the model again, r = 0.01 can be used to change our manipulate our required value, if r is too high an effect occurs
# The best value of r is required, check the next steps after regularization ===> Tuning the model
x_val = prepare_x(df_val)    # prepare the model with the validation dataset
y_pred = w0 + x_val.dot(w)    # predict the validation data

RMSE6 = rmse(y_val, y_pred)
print(RMSE6)  # This regularizeed value is better than the previous ones





# Tuning the Model: Best value for r parameter for the regularization
for r in [0.0, 0.0000001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
    x_train = prepare_x(df_train)    # training dataset
    w0, w = train_linear_regression_reg(x_train, y_train, r=r)   
    x_val = prepare_x(df_val)    
    y_pred = w0 + x_val.dot(w)   

    RMSE_reg = rmse(y_val, y_pred)
    # print(r, w0, w, RMSE_reg)  # w0 is the bias term


# Training the model on the validation set
r = 0.001 # From the tuning model above r = 0.001 seems like a better fit
x_train = prepare_x(df_train)    # training dataset
w0, w = train_linear_regression_reg(x_train, y_train, r=r)   
x_val = prepare_x(df_val)    
y_pred = w0 + x_val.dot(w)   

RMSE_r = rmse(y_val, y_pred)
# print(RMSE_r)


""" Step """
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
w0, w = train_linear_regression_reg(x_full_train, y_full_train, r=0.001) 
# print(f"The bias weight term is: {w0}")
# print(f"The weights are: {w}")

# Prepare the training dataset
x_test = prepare_x(df_test)    # training dataset  
y_pred = w0 + x_test.dot(w) 
score = rmse(y_test, y_pred)
print(score) # This model is better than the other ones so this will be the final model and will be used to predict the price of the car

# Testing the new model;
df_test_car = df_test.iloc[20].to_dict()   # Lets pretend the car is a new car and we need the price of the car
print(df_test_car)
df_small = pd.DataFrame([df_test_car])
print(df_small)
x_small = prepare_x(df_small)
y_pred = w0 + x_small.dot(w)
print(y_pred[0])   # we need a single value and not an array: This is the log value of the car
print(f"The predicted price of the car: {np.expm1(y_pred[0])}")# This is the predicted price of the car


y_test_car = y_test[20]   # the initial price of the car
print(f"The initial price of the car: {np.expm1(y_test_car)}")# This is the predicted price of the car


















