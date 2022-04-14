# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:39:56 2022

@author: ochie
"""
import pandas as pd
import numpy as np
from sklearn import linear_model
import math
from word2number import w2n

df = pd.read_csv("home_prices2.csv")  # Load data into pandas dataframe
# print(df.head(3))  

""" Check the data set given to determine which method is best suited for solving the problem 
    From the data set provided, there's a linear trend between the prices and, age and area of
    the data, this shows that linear regression can be applied 
    The linear equation will look like ===> price = m1 * area + m2 * bedroom + m3 * age + b
    this equation is similar to y = mx + b, where price is dependent on area, bedroom and age 
    {these are called the FEATURES} and are all independent, m1, m2 and m3 are called coefficients
    and b is the intecept """
    

# There's is a missing value from the data provided, let's take care of that first
""" It's noted from the data, a value from bedrooms is missing... Here let's use the medium for that """
median_bedrooms = math.floor(df.bedrooms.median())   # makes the values an integer bcos bedroom can only be an integer no decimal points
# print(median_bedrooms)


""" Fill in the the NaN values with the median value """
df.bedrooms = df.bedrooms.fillna(median_bedrooms)   # fillns the NaN with the median number
# print(df)

# Train the model
reg = linear_model.LinearRegression()
reg.fit(df[["area", "bedrooms", "age"]], df.price)    # dependent variables ===> area, bedrooms and age while price is the target

# Check the coefficients and intercept: These were explained earlier in this script
# print(reg.coef_)
# print(reg.intercept_)

# Let's predict: Input the dependent variables for the prediction
pred1 = reg.predict([[3000, 3, 15]])
print(pred1)

pred2 = reg.predict([[2500, 4, 5]])
print(pred2)


