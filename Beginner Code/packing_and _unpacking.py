# Indexing through names

names = ['George','Smith','Catherine','Emma']

# for name in range(len(names)):
#     print(name + 1,names[name])


#  Packing and Unpacking
# for name in enumerate(names,start=0):
#     print(name)

for num,name in enumerate(names,start=1):
    print(num,name)