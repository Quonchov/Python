smallest = None
print("Before:", smallest)
for itervar in [3, 41, 12, 9, 74, 15]:
    if smallest is None or itervar < smallest:
        smallest = itervar
        break
    print("Loop:", itervar, smallest)
print("Smallest:", smallest)




smallest = None
print("Before:", smallest)
for small in [3, 41, 12, 9, 74, 15]:
    if small < smallest:
        smallest = small
    print("Loop:", small, smallest)
print("Smallest:", smallest)