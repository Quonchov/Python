# Import Functions
import pandas as pd
import numpy as np

# Question 1
# What's the version of NumPy that you installed? You can get the version information using the __version__ field: np.__version__
" Ans = 1.19.5 "

# Question 2
# What's the version of Pandas?
" Ans = 1.3.2 "

# Reading csv files with pandas
car_data = pd.read_csv("C:\\Users\\ochie\\OneDrive\\Desktop\\Python\\machine_learning_zoomcamp\\HW1\\data.csv")
# print(car_data)

# Question 3: Average price of BMW cars
" Ans = 61546.76347305389 "
mean_car_price = car_data.groupby("Make").MSRP.mean() # Groups the cars make by the average car price
bmw_av_price = mean_car_price["BMW"]  # Gets the average BMW car price
# car_data[car_data.Make == "BMW"].MSRP.mean()  # Alternatively 
print(f"The average price of BMW: {bmw_av_price}")

# Question 4
# Select a subset of cars after year 2015 (inclusive, i.e. 2015 and after). How many of them have missing values for Engine HP?
" Ans = 51"
car_year = car_data[(car_data["Year"] >= 2015)] # Shows a list the of car data from 2015 to recent
missing_val = car_year.isnull().sum()   # Sums the missing values for each car feature
hp_mis_val = car_year["Engine HP"].isnull().sum()
print(f"The number of missing values in Engine HP: {hp_mis_val}") # print the number of missing values for Engine HP

# Question 5
# Calculate the average "Engine HP" in the dataset.
# Use the fillna method and to fill the missing values in "Engine HP" with the mean value from the previous step.
# Now, calcualte the average of "Engine HP" again.
# Has it changed?
# Round both means before answering this questions. You can use the round function for that:
" The answer remains the same"
mean_hp_before = car_data["Engine HP"].mean()  # mean Engine HP
mean_hp_before = round(mean_hp_before)
print(f"Mean HP before: {mean_hp_before}")
car_data["Engine HP"] = car_data["Engine HP"].fillna(value=mean_hp_before)
mean_hp_after = car_data["Engine HP"].mean()
mean_hp_after = round(mean_hp_after)
print(f"Mean HP after: {mean_hp_after}")

# Question 6
"Ans = 0.032212320677486125"
# Select all the "Rolls-Royce" cars from the dataset.
# Select only columns "Engine HP", "Engine Cylinders", "highway MPG".
# Now drop all duplicated rows using drop_duplicates method (you should get a dataframe with 7 rows).
# Get the underlying NumPy array. Let's call it X.
# Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
# Invert XTX.
# What's the sum of all the elements of the result?
# Hint: if the result is negative, re-read the task one more time

car_rr = car_data[car_data.Make == "Rolls-Royce"] # Rolls-Royce information
rr_details = car_rr[["Make","Engine HP","Engine Cylinders","highway MPG"]] # Rolls-Royce specific information 
# print(rr_details)
rr_details = rr_details.drop_duplicates(subset=["Engine HP","Engine Cylinders","highway MPG"]) # Drops the duplicate values in the data
# print(rr_details)
X = np.array([rr_details["Engine HP"],rr_details["Engine Cylinders"],rr_details["highway MPG"]])
X = rr_details.values
# print(X)


XT = X.transpose() # Transpose of X
# XT = X.T.dot(X) # Aternative for transpose
# print(XT)

XTX = X.dot(XT)  # matrix-matrix multiplication
# print(XTX)
XTX_inv = np.linalg.inv(XTX)  # inverse of the matrix
# print(XTX_inv)
sum_XTX = XTX_inv.sum()
print(f"The sum of elements of the inverse matrix: {sum_XTX}")


# Questions 7
# Create an array y with values [1000, 1100, 900, 1200, 1000, 850, 1300].
# Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
# What's the value of the first element of w?.

y = np.array([1000,1100,900,1200,1000,850,1300])  # Array of y values 
mul_val = XT.dot(XTX_inv)  # Multiplication of the inverse of XTX with the transpose of X
w = y.dot(mul_val) # Normal Equation result
# w = XTX_inv.dot(X.T).dot(y) # Alternative 
# print(w)
print(f"The first value of w: {w[0]}")


















