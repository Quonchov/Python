# # Using random dir()
# import random
# for i in range(10):
#     print(random.randint(1,10))
    
    
    
# # List Comprehension
# number = [random.randint(1,100) for i in range(100)]
# print(number)


# Shuffled Values
# num = [i for i in range(20)]
# random.shuffle(num)
# print(num)


#  Raced Horses
##shuffled = [7, 17, 4, 19, 3, 
##            6, 14, 0, 2, 15, 
##            101, 18, 8, 5, 13, 
##            12, 11, 9, 1, 16,
##            10, 20 ,30 ,40 ,45]
##
##horses = [[], [], [], [], []]
### print(horses)
##
##for i in range(5):
##    for j in range(5):
##        horses[j].append(shuffled.pop())
##print(horses)
##
##for race in horses:
##    race.sort()
##    print(race)



## List Comprehension

##def cube(n):
##    return n**3
##
##for i in range(1,11):
##    print(cube(i))
##
##num_list = range(1,11)
##cube_nums = [cube(x) for x in num_list]
##print(cube_nums)   # This prints out the result in a list format
##


## Generating a combination of list [x,y,z] using list comprehension
##print([[x,y,z] for x in [1,2,3] for y in [4,5,6] for z in [7,8,9]])



##Using a boolean
three_multiples = [x for x in range(10) if x % 3 == 0]
print(three_multiples)
