# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 18:29:36 2022

@author: ochie
"""

# Canada Income Prediction


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

# Predict Canada's per Capita income in 2020

<<<<<<< HEAD
df = pd.read_csv("C:\\Users\\ochie\\codes\\Python\\machine_learning\\class1\\canada_files.csv")
=======
df = pd.read_csv("\\machine_learning\\canada_files.csv")
>>>>>>> 26e39243a407e22732b3a64d279e37eeec95fcb0


""" First: Make a plot for visualization """

# Make a plot to visualize the data
plt.xlabel("year")
plt.ylabel("Income per Capita(USD)")
plt.scatter(df.year, df.income, color="red", marker="+")


""" Use Linear Regression to train the model and set up prediction"""
reg = linear_model.LinearRegression()
reg.fit(df[['year']], df.income)

# Predict the year for 2020
pre = reg.predict([[2020]])
print(f"This income for 2020 predictionis {pre}")

""" Plot for the predicted data """
# Let's see the prediction using plots
plt.xlabel("year", fontsize=20)
plt.ylabel("income", fontsize=20)
plt.scatter(df.year, df.income, color="red", marker="+")
plt.plot(df.year, reg.predict(df[["year"]]), color="blue")
<<<<<<< HEAD

""" Answer """
# [41288.69409442]
=======
>>>>>>> 26e39243a407e22732b3a64d279e37eeec95fcb0
