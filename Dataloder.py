import os

path = 'data/names/'
for l in os.listdir(path):
    print(l)
    with open(path + l) as d:
        pass