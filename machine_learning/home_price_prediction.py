# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 16:59:42 2022

@author: ochie
"""

# Home Price Predictions

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model


# import home prices in the pandas dataframe
df = pd.read_csv("C:\\Users\\ochie\\machine_learning\\home_prices.csv")
# print(df)

# Make a plot to visualize the data
# %matplotlib inline
# plt.xlabel("area(sqr ft)")
# plt.ylabel("price(USD)")
# plt.scatter(df.area, df.price, color="red", marker="+")

""" From the plot a linear regression will be better """

# Using linear regression from sklearn
reg = linear_model.LinearRegression()   # create an instance
reg.fit(df[["area"]], df.price)     # fit the data

# The regression is ready to predic the information
prediction = reg.predict([[3300]])
# print(prediction)

# Let's see the prediction using plots
plt.xlabel("area", fontsize=20)
plt.ylabel("price", fontsize=20)
plt.scatter(df.area, df.price, color="red", marker="+")
plt.plot(df.area, reg.predict(df[["area"]]), color="blue")


# Now let's predict the price of houses using the areas provided
df_new = pd.read_csv("C:\\Users\\ochie\\machine_learning\\area.csv")
# print(df_new)

# Predict the prices using linear regression
price_new = reg.predict(df_new)
df_new['price_new'] = price_new   # Create a column title for the new prices
# print(price_new)
# print(df_new)


# Let's see the new prediction using plots
plt.xlabel("area", fontsize=20)
plt.ylabel("price", fontsize=20)
plt.scatter(df_new.area, df_new.price_new, color="red", marker="+")
plt.plot(df_new.area, reg.predict(df_new[["area"]]), color="blue")

df_new.to_csv("C:\\Users\\ochie\\machine_learning\\prediction.csv", index=False)  # exports the values predicted