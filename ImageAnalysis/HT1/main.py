from PIL import Image
import numpy as np
from math import pow, sqrt, floor, ceil
import os


ZODIACS = ['Aquarius', 'Aries', 'Cancer', 'Capricorn', 'Gemini', 'Leo', 'Libra', 'Pisces', 'Sagittarius', 'Scorpio', 'Taurus', 'Virgo']
TEST_FILES = ['test.bmp', 'test2.bmp']
PICS = []
PIC_SIDE = 10
EPS = 0.000001

def get_neighbors(row, col):
    ans = []
    if(row + 1 < PIC_SIDE):
        ans.append([row + 1, col])
        if(col + 1 < PIC_SIDE):
            ans.append([row + 1, col + 1])
        if(col - 1 >= 0):
            ans.append([row + 1, col - 1])

    if(row - 1 >= 0):
        ans.append([row - 1, col])
        if(col + 1 < PIC_SIDE):
            ans.append([row - 1, col + 1])
        if(col - 1 >= 0):
            ans.append([row - 1, col - 1])

    if(col + 1 < PIC_SIDE):
        ans.append([row, col + 1])

    if(col - 1 >= 0):
        ans.append([row, col - 1])

    return ans


def modificate(arr):
    ans = np.zeros_like(arr, dtype=float)
    for i in range(PIC_SIDE):
        for j in range(PIC_SIDE):
            if(arr[i][j] == 1):
                ans[i][j] += 1.0
                neibs = get_neighbors(i, j)
                for point in neibs:
                    ans[point[0]][point[1]] += 0.5
    return ans


cwd = os.getcwd()

def distance(pic1, pic2):
    ans = 0.0
    for i in range(PIC_SIDE):
        for j in range(PIC_SIDE):
            ans += pow(pic1[i][j] - pic2[i][j], 2)
    return sqrt(ans)

for i in range(len(ZODIACS)):
    img = Image.open(cwd + "\\pic\\" + ZODIACS[i] + ".bmp")
    arr = np.array(img)
    PICS.append(modificate(arr))

def normalize_pic(pic):
    n = len(pic)
    m = len(pic[0])
    result1 = np.zeros([PIC_SIDE, m])
    if(n > PIC_SIDE):
        step = float(n) / float(PIC_SIDE)
        for j in range(m):
            lb, rb = EPS, step - EPS
            for i in range(PIC_SIDE):
                res = 0.0
                l = floor(lb)
                r = ceil(rb)
                res += (1.0 + l - lb) * pic[l][j]
                res += (rb + 1.0 - r) * pic[r - 1][j]
                if(r - 2 > l):
                    res += pic[r - 2][j]
                lb += step
                rb += step
                result1[i][j] = res
    else:
        result1 = pic
    result = np.zeros([PIC_SIDE, PIC_SIDE])
    step = float(m) / float(PIC_SIDE)
    for i in range(PIC_SIDE):
        lb, rb = EPS, step - EPS
        for j in range(PIC_SIDE):
            res = 0.0
            l = floor(lb)
            r = ceil(rb)
            res += (1.0 + l - lb) * result1[i][l]
            res += (rb + 1.0 - r) * result1[i][r - 1]
            if (r - 2 > l):
                res += result1[i][r - 2]
            lb += step
            rb += step
            result[i][j] = res
    return result


for file in TEST_FILES:
    img = Image.open(cwd + "\\pic\\" + file)
    arr1 = np.array(img)
    arr = np.zeros([PIC_SIDE, PIC_SIDE])
    for i in range(len(arr1)):
        for j in range(len(arr1[i])):
            if (arr1[i][j][0] == 0):
                arr[i][j] = 1.0
 #   print(arr)
    mod_img = modificate(arr)
  #  mod_img = normalize_pic(mod_img)
    min_dist = 1239.0
    min_ind = -1
    for i in range(len(ZODIACS)):
            curr_dist = distance(mod_img, PICS[i])
            if (min_dist > curr_dist):
                min_dist = curr_dist
                min_ind = i
    print("Picture from file " + file + " is most similar to " + ZODIACS[min_ind] + ", distance = " + str(min_dist) +".")