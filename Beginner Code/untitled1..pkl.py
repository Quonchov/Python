# Importing time while converting to seconds

import time, glob

timer = time.asctime().replace(' ','_') + '.pkl'

print(timer)

files = glob.glob('*.pkl')
print(files)