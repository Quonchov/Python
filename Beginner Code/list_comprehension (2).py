# List Comprehension

y = ['George',30,'Sam','Cat',60,70]

# name = [False for i in y if str(i).isnumeric()]
# print(name)

name = [False if str(i).isnumeric() else True for i in y]
print(name)

