# Sample of loop

numbers = [1,2,3,-10,4,-2,-9,-6,10,40]

# def Pos_Neg(x):
#     if x == 0:
#         print(f'{number} is zero')
#     else:
#         if x < 0:
#             print(f'{number} is a Negative Value')
#         else:
#             print(f'{number} is a Positive Value')

# for number in numbers:
#     Pos_Neg(number)
    
    
#  Number Triangle Creation
for row in range(1,21):
    # print(row)
    for col in range(1,row+1):
        # print('{:3}'.format(col),end='')
        print(f'{col:3}',end='')
    print()

#  While Loop
# number = int(input("Enter your date of birth: "))

# while number >= 0:
#     print(number)
#     number -= 1
    # print(number)
    
# List Comprehension
# pos_val = [i for i in numbers if i > 0]
# print(pos_val)

# neg_val = [j for j in numbers if j < 0]
# print(f'THe negative values are thus: {neg_val}')