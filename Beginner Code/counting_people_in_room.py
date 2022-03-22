## Counting the number of people in the room in pairs

# Initialize the number of people as zero
# List the number of people
# For a pair of individual add 2
# if there is a remainder add that to the count



i = o
j = 0

n = int(input("How many people are in the room: "))

for n % 2 == 0:
    i = i + 2
print(f"There are {i/2} pair(s) of people")

if n != 0:
    j = i + 1
print(f"There are {i/2} pair(s) of people and {i/2-1} people")



