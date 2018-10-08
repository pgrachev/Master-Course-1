import random

FILENAME = 'chips2.txt'
N_POIBTS = 118
CENTRE = 0.4
N_DIGITS = 5



f = open(FILENAME, 'w')

for label in range(2):
    sign = 2 * label - 1
    for i in range(N_POIBTS // 2):
        x = round(sign * random.normalvariate(CENTRE, CENTRE), N_DIGITS)
        y = round(sign * random.normalvariate(CENTRE, CENTRE), N_DIGITS)
        f.write(str(x) + ',' + str(y) + ',' + str(label) + '\n')


