""" This code teaches how to save a trained model for reuse """

# Home Price Predictions

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.externals import joblib
import pickle


# import home prices in the pandas dataframe
df = pd.read_csv("home_prices.csv")
print(df.head(3))   # Visulaize the first three items on the data
# print(df)

# Make a plot to visualize the data
# %matplotlib inline  ===> for ipnyb files or jupyter notes
# plt.xlabel("area(sqr ft)")
# plt.ylabel("price(USD)")
# plt.scatter(df.area, df.price, color="red", marker="+")

""" From the plot a linear regression will be better """

# Using linear regression from sklearn
reg = linear_model.LinearRegression()   # create an instance for the linear regression
reg.fit(df[["area"]], df.price)     # fit the data

# The regression is ready to predic the information
prediction = reg.predict([[3300]])      # Shows the prediction for house with an area of 3300
print(prediction)

# Let's see the prediction using plots
# plt.xlabel("area", fontsize=20)
# plt.ylabel("price", fontsize=20)
# plt.scatter(df.area, df.price, color="red", marker="+")
# plt.plot(df.area, reg.predict(df[["area"]]), color="blue")


# Now let's predict the price of houses using the areas provided
df_new = pd.read_csv("area.csv")    # Set of new data provided for prediction
# print(df_new)

# Predict the prices using linear regression
price_new = reg.predict(df_new)
df_new['price_new'] = price_new   # Create a column title for the new prices
# print(price_new)
# print(df_new)


# Let's see the new prediction using plots
# plt.xlabel("area", fontsize=20)
# plt.ylabel("price", fontsize=20)
# plt.scatter(df_new.area, df_new.price_new, color="red", marker="+")
# plt.plot(df_new.area, reg.predict(df_new[["area"]]), color="blue")

df_new.to_csv("prediction.csv", index=False)  # exports the values predicted


""" Saving as pickle and loading """

with open("model_pickle", "wb") as f:
    pickle.dump(reg,f)
    
with open("model_pickle", "rb") as f:
    mp = pickle.load(f)   # mp is the trained saved model and can be loaded as an instance 
    
print(mp.predict([[3300]]))

""" You can also use joblib for the same job, but with large numpy arrays... joblib is preferable """

# Dump your job using joblib
joblib.dump(reg, "model_joblib")

# Load the file
mj = joblib.load("model_joblib")
mj.predict([[3300]])