##  Fizz Buzz Quiz

""" for a range of number from 1 to 100 print in a new line, print Fizz for multiple of 3 and Buzz for a multiple of 5,
    then print FizzBuzz for a multiple of 3 and 5 """

##def fizz_buzz(num):
##    for x in num:
##        if x % 3 == 0:
##            if x % 5 == 0:
##                print('Fizzbuzz')
##            else:
##                print('Fizz')
##        elif x % 5 == 0:
##            print('Buzz')
##        else:
##            print(x)
##
##for i in fizz_buzz(range(1,100)):
##    print(i)
####    print(f'{fizz_buzz(i)}/n')
##    



def fizz_buzz(x):
##    for x in num:
    if x % 3 == 0:
        if x % 5 == 0:
            print('Fizzbuzz')
        else:
            print('Fizz')
    elif x % 5 == 0:
        print('Buzz')
    else:
        print(x)
##
##for i in fizz_buzz(range(1,100)):
##    print(i)
##    print(f'{fizz_buzz(i)}/n')
    
print(fizz_buzz(3))
