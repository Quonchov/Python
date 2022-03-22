#  Machine learning for Classification: Churn (possible to leave) Prediction

""" The project aims to identify customers that are likely to churn or stoping to use a service. Each customer has a score associated with the probability of churning. Considering this data, the company would send an email with discounts or other promotions to avoid churning.
The ML strategy applied to approach this problem is binary classification, which for one instance can be expressed as: g(xi) = yi
In the formula, yi is the model's prediction and belongs to {0,1}, being 0 the negative value or no churning, and 1 the positive value or churning. The output corresponds to the likelihood of churning.
In brief, the main idea behind this project is to build a model with historical data from customers and assign a score of the likelihood of churning. """

import pandas as pd
import numpy as np


#  create a visualization process graphically
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline  # Ensures the plots are displayed in a notebook
""" !wget - Linux shell command for downloading data """
""" To Know !wget - Linux shell command for downloading data
pd.read.csv() - read csv files
df.head() - take a look of the dataframe
df.head().T - take a look of the transposed dataframe
df.columns - retrieve column names of a dataframe
df.columns.str.lower() - lowercase all the letters
df.columns.str.replace(' ', '_') - replace the space separator
df.dtypes - retrieve data types of all series
df.index - retrive indices of a dataframe
pd.to_numeric() - convert a series values to numerical values. The errors=coerce argument allows making the transformation despite some encountered errors.
df.fillna() - replace NAs with some value
(df.x == "yes").astype(int) - convert x series of yes-no values to numerical values.

Classes, functions, and methods:
train_test_split - Scikit-Learn class for splitting datasets. Linux shell command for downloading data. The random_state argument set a random seed for reproducibility purposes.
df.reset_index(drop=True) - reset the indices of a dataframe and delete the previous ones.
df.x.values - extract the values from x series
del df['x'] - delete x series from a dataframe

Functions and methods:

df.isnull().sum() - retunrs the number of null values in the dataframe.
df.x.value_counts() returns the number of values for each category in x series. The normalize=True argument retrieves the percentage of each category. In this project, the mean of churn is equal to the churn rate obtained with the value_counts method.
round(x, y) - round an x number with y decimal places
df[x].nunique() - returns the number of unique values in x series

"""


""" Step 1: Churn Prediction Project"""
df = pd.read_csv("C:\\Users\\ochie\\OneDrive\\Desktop\\Python\\machine_learning_zoomcamp\\Week 3\\WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()
df.head().T # Looks at all the columns while transposing the dataframe

# First change all alphabets to lower case
df.columns = df.columns.str.lower().str.replace(" ","_") # Replace the spaces in the columns with underscores
categorical_columns = list(df.dtypes[df.dtypes == "object"].index)  # for the columns with strings ("objects") create a list for it

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(" ","_")


""" Step 2: Data Preparation """
df.head().T # Now the dataset is more uniform

# Some of the columsn have strings and some have int but are meant tp be strings, check out this numbers and convert accordingly
# E.g df.totalcharges
tc = pd.to_numeric(df.totalcharges, errors="coerce")  # the coerce ignores the errors
df[tc.isnull()][["customerid","totalcharges"]]  # Shows me the missing values
df.totalcharges = pd.to_numeric(df.totalcharges, errors="coerce")
df[tc.isnull()][["customerid","totalcharges"]]  # Shows me the missing values
df.totalcharges = df.totalcharges.fillna(0)    # Fills the missing values with 0, might b=not be the best approach, because it might have been left out on purpose

# Look at the churn variable
df.churn.head()   # Looks at the churn variable
(df.churn == "yes").head()  # if the value is yes it becomes True if no it becomes false
df.churn = (df.churn == "yes").astype(int)  # Replaces the True and false to a number 1 and 0 using astype(int)


""" Step 3: Setting Up the Validation Framework"""

from sklearn.model_selection import train_test_split

train_test_split    # We can see the test and train sizes from this command, thereby spliting the data into two parts:= training and testing then we need to specify the size we need. However, this command splits the dataset into two parts the test and training dataset
df_full_train, df_test = train_test_split(df, test_size=0.2,random_state=1) # The test size is split to 20% telling us how large we want the dataset to be, here and random_state controls the randomness of the training and testing indoces produced
len(df_full_train), len(df_test)   # checks the length of the datasets, spliting the dataset above gives us the train set and test set which 80% and 20% respectively
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1) # 0.25 (25%) is the amount of validation dataset  
len(df_train), len(df_val), len(df_test)  # lengths of the datasets shared into train, test and validation.

"""  
   Understanding sklearn.model_selection Examples
    --------
import numpy as np
from sklearn.model_selection import train_test_split
X, y = np.arange(10).reshape((5, 2)), range(5)
X
    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7],
           [8, 9]])
list(y)
    [0, 1, 2, 3, 4]
    
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
    ...
X_train
    array([[4, 5],
           [0, 1],
           [6, 7]])
y_train
    [2, 0, 3]
X_test
    array([[2, 3],
           [8, 9]])
y_test
    [1, 4]
    
train_test_split(y, shuffle=False)
    [[0, 1, 2], [3, 4]]
    
"""

df_train = df_train.reset_index(drop=True)  # shuffling the dataset does not have an effect on the data or result instead it shows how neat and arrange the code looks
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Get the y_datasets
y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

# Delete the target variable to avioid using it for the training data
del df_train["churn"]
del df_val["churn"]
del df_test["churn"]


""" Step 4: Expolartory Data Analysis (EDA) 
The EDA for this project consisted of:

Checking missing values
Looking at the distribution of the target variable (churn)
Looking at numerical and categorical variables   """

# We will use the full dataset for this part

# Look at the distibution of thr churn value, how many users are churning users and how many are not
# churncount = df_full_train.churn.value_counts()
# print(churncount)

# # Look at the percentage distribution of the churning and non churnong users
# churncount1 = df_full_train.churn.value_counts(normalize=True)
# print(churncount1)  # The values from this result is the churn rate

# #  Calculate the mean of the churn rate
# global_churn_rate = df_full_train.churn.mean()   # Since this is using only 0 and 1 we get a mean of 1 with respect to the datat set
# print(round(global_churn_rate, 2))

# Look at the datatypes 
df_full_train.dtypes    # We are looking for data on the dataset that has values: neglect the seniorcitizen, use tenure, monthlycharges and totalcharges

numerical = ["tenure", "monthlycharges", "totalcharges"]
categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
       'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']  # Drop the customerid

# Check for the number of uniques variables
df_full_train[categorical].nunique()

""" Step 5: Features Importance: churn rate and risk ratio """
""" 
Churn rate: Difference between mean of the target variable and mean of categories for a feature. If this difference is greater than 0, it means that the category is less likely to churn, and if the difference is lower than 0, the group is more likely to churn. The larger differences are indicators that a variable is more important than others.

Risk ratio: Ratio between mean of categories for a feature and mean of the target variable. If this ratio is greater than 1, the category is more likely to churn, and if the ratio is lower than 1, the category is less likely to churn. It expresses the feature importance in relative terms.

Functions and methods:
 
df.groupby('x').y.agg([mean()]) - returns a datframe with mean of y series grouped by x series
display(x) displays an output in the cell of a jupyter notebook. """


#  Look at the churn rate within different groups
churn_female = df_full_train[df_full_train.gender == "female"].churn.mean()
# print(churn_female)

churn_male = df_full_train[df_full_train.gender == "male"].churn.mean()
# print(churn_male)

# Compare with the global churn
global_churn_rate = df_full_train.churn.mean()
# print(global_churn_rate)

#  Look at customers that lives with partners and those that do not
churn_partners = df_full_train.partner.value_counts()
# print(churn_partners)

churn_partners_yes = df_full_train[df_full_train.partner == "yes"].churn.mean()
# print(churn_partners_yes)  # The result here can be compared with the global churn rate and it can be noted that people who lives with partners has a churn rate of 20%

churn_partners_no = df_full_train[df_full_train.partner == "no"].churn.mean()
# print(churn_partners_no)  # The result here can be compared with the global churn rate and it can be noted that people who lives with partners has a churn rate of 20%

# Difference between global churn and those who has a partner
diff_churn1 = global_churn_rate - churn_partners_yes
# print(diff_churn1)

# Difference between global churn and those with no partner
diff_churn2 = global_churn_rate - churn_partners_no
# print(diff_churn2)

#  Risk Ratio: Risk = Group churn rate / Global churn rate:  If it is less than one they are less likely to churn but if it is greater than one they are most likely to churn

rr1 = churn_partners_no / global_churn_rate
# print(rr1)

rr2 = churn_partners_yes / global_churn_rate
# print(rr2)


df_group = df_full_train.groupby(by='gender').churn.agg(['mean'])
df_group['diff'] = df_group['mean'] - global_churn_rate
df_group['risk'] = df_group['mean'] / global_churn_rate
df_group

#  Sequel Query to Pandas
from IPython.display import display   # Use this for display, because our loop will not display the details

for c in categorical:
    # print(c)
    df_group = df_full_train.groupby(c).churn.agg(["mean","count"])
    df_group["diff"] = df_group["mean"] - global_churn_rate
    df_group["risk"] = df_group["mean"] / global_churn_rate
    display(df_group)
    # print()
    # print()


""" Step 6: Feature Importance: Mutual Information """ # Tells us which information is important or not
""" Mutual information is a concept from information theory, which measures how much we can learn about 
one variable if we know the value of another. In this project, we can think of this as how much do we learn 
about churn if we have the information from a particular feature. So, it is a measure of the importance of a 
categorical variable.

Classes, functions, and methods:

mutual_info_score(x, y) - Scikit-Learn class for calculating the mutual information between the x target variable and y feature.
df[x].apply(y) - apply a y function to the x series of the df dataframe.
df.sort_values(ascending=False).to_frame(name='x') - sort values in an ascending order and called the column as x. """


from sklearn.metrics import mutual_info_score     # Sklearn implements mutual information

mutual_score = mutual_info_score(df_full_train.contract, df_full_train.churn) # Tells us how much we learn about the churn by observing the contract variable and vise versa
# print(mutual_score)

# df_full_train.apply(): This apply a function to a piece of information provided

def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)

mutual_info = df_full_train[categorical].apply(mutual_info_churn_score)   # Use categorical variable to tell us the information we needed

# Sorts the mutual information
mutual_info = mutual_info.sort_values(ascending=False)
# print(mutual_info)



""" Step 7: Feature Importance - Correlation """ # For measuring numerical variables; Way of measuring the degree of dependency between two variables

""" Correlation coefficient measures the degree of dependency between two variables. This value is negative 
if one variable grows while the other decreases, and it is positive if both variables increase. Depending on 
its size, the dependency between both variables could be low, moderate, or strong. It allows measuring the importance 
of numerical variables.

Functions and methods:

df[x].corrwith(y) - returns the correlation between x and y series."""

corr = df_full_train[numerical].corrwith(df_full_train.churn)
# print(corr)

two_months_churn = df_full_train[df_full_train.tenure <= 2].churn.mean()   # People who stayed in the company in less than or two months
# print(two_months_churn)

monthlycharges_churn = df_full_train[df_full_train.monthlycharges <= 20].churn.mean()   # People who pay less than or equal to $20 per month 
# print(monthlycharges_churn)



""" Step 8: One Hot Encoding """

""" One-Hot Encoding allows encoding categorical variables in numerical ones. This method represents 
each category of a variable as one column, and a 1 is assigned if the value belongs to the category or 0 otherwise.

Classes, functions, and methods:

df[x].to_dict(oriented='records') - convert x series to dictionaries, oriented by rows.
DictVectorizer().fit_transform(x) - Scikit-Learn class for converting x dictionaries into a sparse matrix, 
and in this way doing the one-hot encoding. It does not affect the numerical variables.
DictVectorizer().get_feature_names() - returns the names of the columns in the sparse matrix. """

from sklearn.feature_extraction import DictVectorizer # DictVectorizer converts a dictionary to a vector

x = df_train[["gender", "contract"]].iloc[:]
# print(x)

v = df_train[["gender", "contract"]].iloc[:10].to_dict(orient="records")
dv = DictVectorizer(sparse=False)
dv.fit(v)
# dv.get_feature_names()  # Gets the feature names
test_dv = dv.transform(v)
# print(dv)

train_dicts = df_train[categorical + numerical].to_dict(orient="records")
# print(train_dicts[0])
# dv.fit(train_dicts)
# print(dv.get_feature_names())  # Gets the feature names
# train_dicts = dv.transform(train_dicts)
x_train = dv.fit_transform(train_dicts)   # It first fits the data before transforming
# print(x_train)
x_train.shape    # gives the shape of the trained dataset

""" Don't fit the validation dataset """
val_dicts = df_val[categorical + numerical].to_dict(orient="records")
x_val = dv.transform(val_dicts)



""" Step 9: Logistics Regression: Solves binary classification """

""" In general, supervised models follow can be represented with this formula:
g(xi) = yi
Depending on what is the type of target variable, the supervised task can be regression 
or classification (binary or multiclass). Binary classification tasks can have negative (0)
or positive (1) target values. The output of these models is the probability of xi belonging 
to the positive class.

Logistic regression is similar to linear regression because both models take into account the 
bias term and weighted sum of features. The difference between these models is that the output 
of linear regression is a real number, while logistic regression outputs a value between zero and
one, applying the sigmoid function to the linear regression formula.

g(xi) = Sigmoid(w0 + w1x1 + w2x2 + ... + wnxn)
Sigmoid = 1/(1+exp(-z))

In this way, the sigmoid function allows transforming a score into a probability. """ 

"Compare logistic regression with linear regression"

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-7, 7, 51)
#z = sigmoid(z)
# plt.plot(z,sigmoid(z))


def linear_regression(xi):
    result = w0
    
    for j in range(len(w)):
        result = result + xi[j] * w[j]
        
    return result


def logistic_regression(xi):
    score = w0
    
    for j in range(len(w)):
        score = score + xi[j] * w[j]
        
    result = sigmoid(score)
    return result



""" Step 10: Training Logistics Regression with Scikit-Learn """

""" This video was about training a logistic regression model with Scikit-Learn, 
applying it to the validation dataset, and calculating its accuracy.

Classes, functions, and methods:

LogisticRegression().fit_transform(x) - Scikit-Learn class for calculating the logistic regression model.
LogisticRegression().coef_[0] - returns the coeffcients or weights of the LR model
LogisticRegression().intercept_[0] - returns the bias or intercept of the LR model
LogisticRegression().predict[x] - make predictions on the x dataset
LogisticRegression().predict_proba[x] - make predictions on the x dataset, and returns two columns with 
their probabilities for the two categories - soft predictions """

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)


model.intercept_[0]   # These are the coefficients also called the bias (Bias term is what we assume for the customer without knowing them)
model.coef_[0].round(3)   # These are the weights
model.predict(x_train)  # Predicts the numbers in zeros and one as no-churn and churn respectively - Hard predictions since we don't know the probability

y_pred = model.predict_proba(x_val)[:, 1] # Takes only the probability of churning # Known as the soft predictions - gives two columns, with column 1 being the negative part (prbability of not churning) and column two being the positive and important part of the dataset and probability of churning
# y_pred = model.predict_proba(x_train)[:, 1]
churn_decision = (y_pred >= 0.5)   # true means customer is churning and false means otherwise. This is the decision threshold that determines if the customer will churn

df_val[churn_decision].customerid   # customers likely to churn while seleting the roles for which the churn_decision is true

churn_decision.astype(int)

(y_val == churn_decision).mean()  # shows how many percentage of the prediction matches here its 80% correct

df_pred = pd.DataFrame()
df_pred["probability"] = y_pred
df_pred["predicition"] = churn_decision.astype(int)
df_pred["actual"] = y_val

df_pred['correct'] = df_pred.predicition == df_pred.actual   # Anytime the prediction is correct we have a True on the column and False otherwise

df_pred.correct.mean()



""" Step 11: Model Interpretation """
""" This video was about the interpretation of coefficients, and training a model with fewer features.

In the formula of the logistic regression model, only one of the one-hot encoded categories is multiplied by 1, 
and the other by 0. In this way, we only consider the appropriate category for each categorical feature.

Classes, functions, and methods:

zip(x,y) - returns a new list with elements from x joined with their corresponding elements on y """

dict(zip(dv.get_feature_names(), model.coef_[0].round(3))) # Turn to a dictionary

# Train a small model
small = ["contract", "tenure", "monthlycharges"]
df_train[small].iloc[:10].to_dict(orient="records") # for viewing the dataset

dicts_train_small = df_train[small].to_dict(orient="records")
dicts_val_small = df_val[small].to_dict(orient="records")

dv_small = DictVectorizer(sparse=False)
dv_small.fit(dicts_train_small)

dv_small.get_feature_names()

# Training the model
x_train_small = dv_small.transform(dicts_train_small) 
model_small = LogisticRegression()
model_small.fit(x_train_small, y_train)

# Bias Terms of the model
w0 = model_small.intercept_[0]
print(w0)

w = model_small.coef_[0]
w = w.round(3)

# Get the required information which is the weights
dict(zip(dv_small.get_feature_names(), w))
# sigmoid(_)
ss = sigmoid(-2.47 + 0.97 + 50 * 0.027 + 5 * (-0.036))   # Using the weights above for a monthly contract 50 is the monthly fee, and 5 is the number of months spent in the company
# print(ss)


""" Step 12: Using the model """

""" We trained the logistic regression model with the full training dataset (training + validation), considering 
numerical and categorical features. Thus, predictions were made on the test dataset, and we evaluate the model using 
the accuracy metric.

In this case, the predictions of validation and test were similar, which means that the model is working well."""


dicts_full_train = df_full_train[categorical + numerical].to_dict(orient="records")
# print(dicts_full_train[:3])

dv = DictVectorizer(sparse=False)

# Train the dataset
x_full_train = dv.fit_transform(dicts_full_train)
y_full_train = df_full_train.churn.values
model = LogisticRegression()
model.fit(x_full_train, y_full_train)

# Test dataset
dicts_test = df_test[categorical + numerical].to_dict(orient="records")
x_test = dv.transform(dicts_test)
y_pred_new = model.predict_proba(x_test)[:,1]
churn_decision_new = (y_pred >= 0.5)
(churn_decision_new == y_test).mean()   # gives 81%, which was slightly more than the validation dataset prediction and the accurancy is small meaning the model is good.

customer = dicts_test[10] # The 10th customer on the list
print(customer)

x_small_new = dv.transform([customer])
# x_small_new.shape  # one customer with 45 features

customer1 = model.predict_proba(x_small_new)[0,1]
print(customer1)   # The result shows that the probability of the customer churning is low.

print(y_test[10])  # The result depicts the prediction was correct because the customer was not going to churn, a result of 0 means no churn and 1 means churn







