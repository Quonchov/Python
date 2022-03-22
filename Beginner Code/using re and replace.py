# Replacing space and colons with underscores in the 'file' document
file = 'Thurs Feb 1990:21:01'

# change = " :" # Changing spaces and colon
# for char in change:
#     file = file.replace(char,"_")
# print(file)


# regex, RE- Regular Expressions

import re

file = re.sub("[ :]","_",file)

print(file)