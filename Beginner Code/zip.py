# Zip

names = ['George','Sam','Catherine','Daniel']
scores = [80,98,100,80]

# grades = dict(zip(names,scores))

# for names,scores in grades.items():
#     print(f'{name} had a total score of {scores}')
    
# Packing 
grades = list(zip(names,scores))
print(grades)


# Unpacking or Unzipping
new_students, scores = zip(*grades)
print(new_students,scores)



# Using strings and help

x = 'help'
print(x.center(50,'#'))   # center() positions the string at the center with offsets by the side