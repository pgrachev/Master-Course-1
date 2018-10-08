import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from math import exp
from sklearn.svm import SVC

C = 10.0
GAMMA = 0.4
OPT_K = 100
EPS = 0.001
TEST_PERCENT = 0.2
total_cnt = 0

def kernel (a, b):
    return a[0] * b[0] + a[1] * b[1]

def shift(x):
    return 2 * x - 1

def reverse_shift(x):
    return int((x + 1) / 2)

class SVM:
    n_points = 0
    points = []
    lambdas = []
    w = [None, None]
    b = None
    def fit_transform(self, points):
        self.points = points[:]
        self.n_points = len(points)
        for i in range(self.n_points):
            self.points[i][2] = shift(self.points[i][2])
        self.lambdas = [0.0 for _ in points]
        opt_number = self.n_points * OPT_K
        for i in range(opt_number):
            ind1 = random.randint(0, self.n_points - 1)
            ind2 = random.randint(0, self.n_points - 1)
            while(ind2 == ind1):
                ind2 = random.randint(0, self.n_points - 1)
            self.optimize(ind1, ind2)

        self.w = [0.0, 0.0]

        for i in range(self.n_points):
            self.w[0] = self.w[0] + self.lambdas[i] * self.points[i][2] * self.points[i][0]
            self.w[1] = self.w[1] + self.lambdas[i] * self.points[i][2] * self.points[i][1]

        def cnt_err(b):
            ans = 0
            for p in self.points:
                if(int(shift(int(kernel(self.w, p) - b > 0))) != int(p[2])):
                    ans = ans + 1
            return ans

        lb = -10.0
        rb = 10.0
        step = 0.1
        self.b = lb
        minres = cnt_err(lb)
        curr_b = lb + step
        while(curr_b < rb):
            if(cnt_err(curr_b) < minres):
                self.b = curr_b
                minres = cnt_err(curr_b)
            curr_b = curr_b + step


    def optimize(self, ind1, ind2):
        K1 = 0.5 * kernel(self.points[ind1], self.points[ind1])
        K2 = 0.5 * kernel(self.points[ind2], self.points[ind2])
        K3, K4, K5 = 0, 0, 0
        for i in range(self.n_points):
            if(i != ind1):
                K3 = K3 + self.points[i][2] * self.lambdas[i] * kernel(self.points[ind1], self.points[i])
            if(i != ind2):
                K4 = K4 + self.points[i][2] * self.lambdas[i] * kernel(self.points[ind2], self.points[i])
                if (i != ind1):
                    K5 = K5 + self.lambdas[i] * self.points[i][2]
        K3 = K3 * 0.5 * self.points[ind1][2]
        K4 = K4 * 0.5 * self.points[ind2][2]
        x0 = [self.lambdas[ind1], self.lambdas[ind2]]

        def f(x):
            return K1 * x[0] * x[0] + K2 * x[1] * x[1] + K3 * x[0] + K4 * x[1] - x[0] - x[1]

        constraints = ({'type': 'eq', 'fun': lambda x: self.points[ind1][2] * x[0] + self.points[ind2][2] * x[1] + K5})
        bounds = ((0, C), (0, C))

        res = minimize(f, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        self.lambdas[ind1] = res.x[0]
        self.lambdas[ind2] = res.x[1]

    def predict(self, p):
        return shift(int(kernel(self.w, p) - self.b > 0))
'''

f = open('chips.txt')
points = []


for line in f:
    inp = line.split(',')
    points.append([float(inp[0]), float(inp[1]), int(inp[2])])
random.seed(7)
random.shuffle(points)
n_points = len(points)
test_cnt = int(n_points * TEST_PERCENT)
test_set = points[0:test_cnt]
points = points[test_cnt:n_points]
nppoints = np.array(points)
npX = nppoints[:, 0:2]
npy = nppoints[:, -1]
svc = SVC(C=C, kernel='poly')
svc.fit(npX, npy)

tstset = np.array(test_set)
npX = tstset[:, 0:2]
npy = tstset[:, -1]
pred = svc.predict(npX)

from sklearn.metrics import accuracy_score, f1_score

print(accuracy_score(pred, npy))

posx, posy, negx, negy = [], [], [], []
for point in points:
    if (point[2] == 1):
        posx.append(point[0])
        posy.append(point[1])
    else:
        negx.append(point[0])
        negy.append(point[1])
plt.plot(posx, posy, 'g^')
plt.plot(negx, negy, 'bs')
plt.axis([-1, 1, -1, 1])

svm = SVM()
svm.fit_transform(points)
lx = []
ly = []
lb = -1.0
rb = 1.0
step = 0.01
w0 = lb
while (w0 < rb):
    w1 = (svm.b - svm.w[0] * w0) / svm.w[1]
    lx.append(w0)
    ly.append(w1)
    w0 = w0 + step


plt.plot(lx, ly, 'r')
plt.show()


CM = [[0, 0], [0, 0]]

for point in test_set:
    i1 = reverse_shift(svm.predict(point))
    i2 = point[2]
    CM[i1][i2] = CM[i1][i2] + 1

print('Confusion matrix: ')
for s in CM:
    print(str(s[0]) + ' ' + str(s[1]))
precision = CM[0][0] / (CM[0][0] + CM[0][1])
recall = CM[0][0] / (CM[0][0] + CM[1][0])
f1m = 2 * precision * recall / (precision + recall)

print('f1 measure: ' + str(f1m))

'''