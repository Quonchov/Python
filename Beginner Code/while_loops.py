scores = list()

while True:
    print("The last student should input 'done' after adding your number...")
    inp = input("What is your score: ").lower()
    if inp == 'done': break
    #values = float(inp)
    scores.append(inp).pop()

average = sum(scores) / len(scores)
print("Average: ", average)



