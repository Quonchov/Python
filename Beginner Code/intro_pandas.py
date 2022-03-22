""" Task 1: Create a DataFrame
Do the following:

Create an 3x4 (3 rows x 4 columns) pandas DataFrame in which the columns are named Eleanor, Chidi, Tahani, and Jason. Populate each of the 12 cells in the DataFrame with a random integer between 0 and 100, inclusive.

Output the following:

the entire DataFrame
the value in the cell of row #1 of the Eleanor column
Create a fifth column named Janet, which is populated with the row-by-row sums of Tahani and Jason.

To complete this task, it helps to know the NumPy basics covered in the NumPy UltraQuick Tutorial."""


import pandas as pd
import numpy as np

# Create the list of names
Names = ["Eleanor", "Chidi", "Tahani", "Jason"]

# Create a 3 by 4 numpy array
Num = np.random.randint(low= 0, high= 101, size= ([3,4]))


#  Create a dataframe
df = pd.DataFrame(data=Num, columns=Names)


print("The entire DataFrame: ")
print(df,"\n")

# Eleanor column for row1
print("Eleanor column for row #1: ")
print(df.iloc[1,0], "\n")
# OR Print the value in row #1 of the Eleanor column.
print("\nSecond row of the Eleanor column: %d\n" % df['Eleanor'][1])

#  Create fifth column "Janet"
df["Janet"] = df["Tahani"] + df["Jason"]
print("Updated dataframe: ")
print(df)
# print(Num)


# Copying an dataFrame
# Create a reference by assigning my_dataframe to a new variable.
print("Experiment with a reference:")
reference_to_df = df

# Print the starting value of a particular cell.
print("  Starting value of df: %d" % df['Jason'][1])
print("  Starting value of reference_to_df: %d\n" % reference_to_df['Jason'][1])

# Modify a cell in df.
df.at[1, 'Jason'] = df['Jason'][1] + 5
print("  Updated df: %d" % df['Jason'][1])
print("  Updated reference_to_df: %d\n\n" % reference_to_df['Jason'][1])

# Create a true copy of my_dataframe
print("Experiment with a true copy:")
copy_of_my_dataframe = my_dataframe.copy()

# Print the starting value of a particular cell.
print("  Starting value of my_dataframe: %d" % my_dataframe['activity'][1])
print("  Starting value of copy_of_my_dataframe: %d\n" % copy_of_my_dataframe['activity'][1])

# Modify a cell in df.
my_dataframe.at[1, 'activity'] = my_dataframe['activity'][1] + 3
print("  Updated my_dataframe: %d" % my_dataframe['activity'][1])
print("  copy_of_my_dataframe does not get updated: %d" % copy_of_my_dataframe['activity'][1])