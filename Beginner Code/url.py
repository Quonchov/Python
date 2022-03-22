#  URL 
''' Here this code shows how to import from the web, then create a txt file
    opened as a text.'''

import urllib.request as url

page = url.urlopen('http://www.textfiles.com/etext/AUTHORS/POE/poe-raven-702.txt')


text = page.read()
text = text.decode()


text = text.split()
peom = {}

for word in text:
    if word not in peom:
        peom[word] = 1
    else:
        peom[word] += 1
print(peom)
        
        
# Sorting the words in the peom
def last(x):
    return x[-1]

# names = ['geo','Jeo','kanu']
# print(sorted(names,key=last,reverse=True))


peom = peom.items()
sort_peom = sorted(peom, key=last, reverse= True) #, reverse=True)
for peom_word in sort_peom[:10]:
    print(peom_word[0], ':', peom_word[1])
# print(sort_peom)
    
    
    
# print(sorted(peom, key = 'word'))  
        
# for key, value in peom.items():
#     if value >= 9:
#         print(f'This {key} can be counted')
#     # else:
#         print(f'{key}s have values less than 9')


#  Creating a txt file
# file = open('raven.txt','w')

# file.write(text)
# file.close()