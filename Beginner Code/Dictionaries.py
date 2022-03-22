#  Dictionaries

# fruits = {"Mango":300, "Oranges":200, "Kiwi":49, "Banana":500}
# # print(fruits['Mango'])

# for fruit,number in fruits.items():
#     if fruits['Mango'] == 100:
#         print(f'I have a total of {number} {fruit} in my bags!!!!')
#     else:
#         print('No fruit is available')


# from string import ascii_lowercase as lower

# key = {}

# for i in range(len(lower)):
#     key[lower[i]] = 1 + i
# print(key)

# Dictionary Zipping

# letters = lower
# num = list(range(1,27))

# alphabets = dict(zip(letters,num))
# print(alphabets)


#  Dictionary Compreshension

students = ['sam','john','tom','dave','cathy']

# dict_students = {student[0].upper(): student for student in students}
dict_students = {len(students): student for student in students}

print(dict_students)
