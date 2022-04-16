# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 18:17:05 2022

@author: ochie
"""
""" This code predicts the salaries of potential employees given their experience, test_score and interview_score """

import pandas as pd
import numpy as np
from sklearn import linear_model
from word2number import w2n
import math

df = pd.read_csv("hiring.csv")  # Load data into pandas dataframe
# print(df.head(3))

# Remove the NaN
median = math.floor(df.test_score.median()) # create a median value from the test_score dataset
# print(median)

df.test_score = df.test_score.fillna(median) # fill the NaN value with the median value
df.experience = df.experience.fillna("zero")  # fill the NaN value in experience with "zero" string

# Convert the experience string numbers to integers, using word2number
df.experience = df.experience.apply(w2n.word_to_num)  # converts the string numbers to integers

# Updated dataframe
print(df)


# Train the model
reg = linear_model.LinearRegression()
reg.fit(df[["experience", "test_score", "interview_score"]], df.salary)

# Check the coefficients and intercept: These were explained earlier in this script
# print(reg.coef_)
# print(reg.intercept_)

# Let's predict for: 
""" 2yrs, 9 test_score and 6 interview_score
    12yrs, 10 test_score and 10 interview_score """

employee1 = reg.predict([[2, 9, 6]])
print(f'The salary of this employee is {employee1}')

employee2 = reg.predict([[12, 10, 10]])
print(f'The salary of this employee is {employee2}')




