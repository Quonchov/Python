# factorial of given number
def fac(n):
    return 1 if (n==1 or n==0) else n * fac(n - 1);

num = 12;
print(f"The Factorial of {num} is", fac(num))