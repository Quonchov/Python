from string import ascii_lowercase as al


abc = ''
for letter in al:
    # abc += letter + "*"
    abc = '*'.join(al)
print(abc)
    
# name = 'George'
# print(name[-1])