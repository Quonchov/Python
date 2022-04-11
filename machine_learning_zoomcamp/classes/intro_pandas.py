# pandas is a library for manipulating tabular data in python

import pandas as pd

# lists of car data
car_info = [
    ['Nissan', 'Stanza', 1991, 138, 4, 'MANUAL', 'sedan', 2000],
    ['Hyundai', 'Sonata', 2017, None, 4, 'AUTOMATIC', 'Sedan', 27150],
    ['Lotus', 'Elise', 2010, 218, 4, 'MANUAL', 'convertible', 54990],
    ['GMC', 'Acadia',  2017, 194, 4, 'AUTOMATIC', '4dr SUV', 34450],
    ['Nissan', 'Frontier', 2017, 261, 6, 'MANUAL', 'Pickup', 32340],
]

columns = [
    'Make', 'Model', 'Year', 'Engine HP', 'Engine Cylinders',
    'Transmission Type', 'Vehicle_Style', 'MSRP'
]

car_df = pd.DataFrame(data=car_info, columns=columns)
print(car_df,"\n")

# to view "make" of the car
car_df_make = car_df.Make
car_df["Make"] # You can also use this method
print("The makes of the cars: ")
print(car_df_make,"\n")

# # to view the "Engine HP" car_df["Engine HP"] is preferred, due to the space
# print("Engine HPs: ")
# print(car_df["Engine HP"])

# #  for multiple views
# print("Some other aspects of the car: ")
# print(car_df[["Make", "Model","MSRP"]])

# # adding new column "No of doors"
# car_df["No. of doors"] = [2,2,4,4,2]
# print(car_df)

# #  to delete a column 
# del car_df["Model"]


# # Indexing: to check for the index
# print("Shows the index: ")
# print(car_df.index)
# print(car_df.loc[[1]]) # displays in horizontal format
# print(car_df.loc[1])  # displays in vertical format
# print(car_df.loc[[1,2]])

# # changing the index from 0-4 to a-e
# car_df.index = ['a','b','c','d','e'] # this changes the car_df.loc[] function above
# # however using car_df.iloc[] solves the problem
# # print(car_df.reset_index(drop=True)) # the drop is optional

# # # dictionary of dataFrame
# # data = [
# #     {
# #         "Make": "Nissan",
# #         "Model": "Stanza",
# #         "Year": 1991,
# #         "Engine HP": 138.0,
# #         "Engine Cylinders": 4,
# #         "Transmission Type": "MANUAL",
# #         "Vehicle_Style": "sedan",
# #         "MSRP": 2000
# #     },
# #     {
# #         "Make": "Hyundai",
# #         "Model": "Sonata",
# #         "Year": 2017,
# #         "Engine HP": None,
# #         "Engine Cylinders": 4,
# #         "Transmission Type": "AUTOMATIC",
# #         "Vehicle_Style": "Sedan",
# #         "MSRP": 27150
# #     },
# #     {
# #         "Make": "Lotus",
# #         "Model": "Elise",
# #         "Year": 2010,
# #         "Engine HP": 218.0,
# #         "Engine Cylinders": 4,
# #         "Transmission Type": "MANUAL",
# #         "Vehicle_Style": "convertible",
# #         "MSRP": 54990
# #     },
# #     {
# #         "Make": "GMC",
# #         "Model": "Acadia",
# #         "Year": 2017,
# #         "Engine HP": 194.0,
# #         "Engine Cylinders": 4,
# #         "Transmission Type": "AUTOMATIC",
# #         "Vehicle_Style": "4dr SUV",
# #         "MSRP": 34450
# #     },
# #     {
# #         "Make": "Nissan",
# #         "Model": "Frontier",
# #         "Year": 2017,
# #         "Engine HP": 261.0,
# #         "Engine Cylinders": 6,
# #         "Transmission Type": "MANUAL",
# #         "Vehicle_Style": "Pickup",
# #         "MSRP": 32340
# #     }
# # ]

# # df_car_dict = pd.DataFrame(data)
# # print(df_car_dict)

# # # using the head method
# # print(car_df.head(2)) # very handy when you have a large dataset, you take a peek at the first two




# #  Element wise operations
# Hp = car_df["Engine HP"] * 2
# print(Hp)


# # logical operations
# year = car_df["Year"] >= 2015 # This gives a boolean

# #  Filtering
# year = car_df[car_df["Year"] >= 2015]   # Returns only True outputs the list of cars after 2015 or equal
# print(year)

# car_df["Make"] == "Nissan"

# # combination of logical expression
# print(car_df[(car_df["Make"] == "Nissan") & (car_df["Year"] >= 2015)])


# # lower functions
# car_df["Vehicle_Style"].str.lower()

# print(car_df.MSRP.describe())  # outputs the numerical details of the car

# # category
# print(car_df.Make.nunique())
# print(car_df.nunique())

# #  for missing values
# car_df.isnull()
# car_df.isnull().sum() # output number of missing values


# # Grouping
# print("Prints Minimum: ")
# print(car_df.groupby("Transmission Type").MSRP.min(),"\n")

# print("Prints Max: ")
# print(car_df.groupby("Transmission Type").MSRP.max())

# #  Getting the Numpy arrays
# print(car_df.MSRP.values)

# # converting the dataframe to dictionary
# print(car_df.to_dict(orient='records'))