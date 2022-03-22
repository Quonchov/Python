x = int(input("Type in the value of x: "))
y = int(input("Type in the value of y: "))
z = int(input("Type in the value of z: "))
n = int(input("Type in the test value: "))

num = [[i,j,k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if (i+j+k) != n]
print(num)

