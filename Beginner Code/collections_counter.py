from collections import Counter as c


##count = c('abcdeabcdabcaba')
##print(count)
##
##
##print(count.most_common(2))


##from collections import Counter
##
##myList = [1,1,2,3,4,5,3,2,3,4,2,1,2,3]
##
##print Counter(myList)
####  Result:  Counter({2: 4, 3: 4, 1: 3, 4: 2, 5: 1})
##
##print Counter(myList).items()
####  Result:  [(1, 3), (2, 4), (3, 4), (4, 2), (5, 1)]
##
##print Counter(myList).keys()
####  Result:  [1, 2, 3, 4, 5]
##
##print Counter(myList).values()
####  Result:  [3, 4, 4, 2, 1]


##Raghu has X number of shoes
##
##he has a list that contains the size of each shoe
##
##There are N number of customers who has agreed to pay Xi of the shoe price provided it fits them.
##
##Compute how much Raghu earned

### Enter your code here. Read input from STDIN. Print output to STDOUT
##import collections
##
##numShoes = int(input())
##shoes = collections.Counter(map(int, input().split()))
##numCust = int(input())
##
##income = 0
##
##for i in range(numCust):
##    size, price = map(int, input().split())
##    if shoes[size]: 
##        income += price
##        shoes[size] -= 1
##
##print(income)

##X = int(input('Number of shoes: '))
##
##shoe_sizes = ('2,3,4,5,6,8,7,6,5,18').replace(',',' ')
##
####shoe_cost = ('100,300,10,
##
##N = int(input('Number of customers: '))
##
##for size in shoe_sizes:
##    print(f'shoe size of {size} for the cost of {cost}')



num_Shoes = int(input('Enter the number of shoes: '))
shoes_size = c(map(int, input('Enter the shoe sizes of customers: ' ).split()))
num_Customers = int(input('How many customers are required: '))

income = 0

for i in range(num_Customers):
    size, price = map(int, input('Enter the shoe size and cost: ').split())
    if shoes_size[size]: 
        income += price
        shoes_size[size] -= 1

print(income)



## Check for split() map(int, input())





