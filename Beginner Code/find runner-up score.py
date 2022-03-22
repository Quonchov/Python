# find the runner-up score

num =[]

n = int(input("Total number of participants: "))

arr = list(map(int,input("Input an array of n numbers: ").split()))
# arr = list(arr)
print(arr)
x = max(arr)
y = min(arr)

for i in range(0,n):
    if arr[i] < x and arr[i] > y:
        y = arr[i]
print(y)    
    
    
    
# Tried this code, might update this later using sort

# for i in arr:
#     num.append(i)
#     num.sort()
#     for j in num:
#         if 
# print(num)


# if 
# print(max(arr)-1)
# 1 2 3 4 

# 57 57 -57 57

# -2 -4 -5 -7

