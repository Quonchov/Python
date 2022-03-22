#  Multiplication Table
# ''' This code calculates the muliplication of input integer numbers'''

# n = int(input('Input a single integer n: ').strip())

# for i in range(1,2):
#     while i <= 10:
#         print(f'{n} x {i} =', n * i)
#         i += 1
# print()
            


#  Multiplication Table
# def times_table():
#     while True:
#         try:
#             x = int(input("Please type in a number: "))
#             for row in range(x+1):
#                 for col in range(x+1):
#                     print(f'{row * col:3}',end=' ')
#                 print()
#         except ValueError:
#             print('Ooops enter a new value:')
#         ques = input('Do you want to try another value? y/n: ').lower()
#         if ques[0] == 'n':  # ques[0] picks on anything that starts with the letter n
#             break
#     print("End of code.")
        
# times_table()

#  Check how long this code can run with a value of 1000

#  Prime Numbers

def prime_number():
    i = int(input("Type in a value to confirm if its a prime number: "))
    # for i in range(2,x):  
    if i / i == 1 and i / 1 == i and i % 2 == 1:
        print(f'{i} is a prime number.')
    else:
        print(f'{i} is not a prime number.')
    
prime_number()

def is_prime(num):
    ''' Returns a True of False result from the input value'''
    for i in range(2,num):
        if num % i == 0:
            return False
    return True

# test_val = list(range(1,26))

# for num in test_val:
#     print(f'{num} is a prime number: {is_prime(num)}')