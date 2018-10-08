import random

MU = 0.0000001
EPS = 0.01
MAX_RAND_WEIGHT = 100.0

def MSE(EXP, RLT):
    N, ans = len(EXP), 0
    for i in range(N):
        ans += (EXP[i] - RLT[i]) ** 2
    return ans / N


class LinearRegressor:
    w = []
    b = None

    def __init__(self):
        self.w = [random.random() * MAX_RAND_WEIGHT, random.random() * MAX_RAND_WEIGHT]
        self.b = random.random() * MAX_RAND_WEIGHT

    def predict(self, X):
        return [v[0] * self.w[0] + v[1] * self.w[1] + self.b for v in X]

    def train(self, X, Y):
        N = len(Y)
        dw = [0, 0]
        db = 0
        pred = self.predict(X)
        err = MSE(Y, pred)
        dw[0] = 2 / N * sum([(pred[j] - Y[j]) * X[j][0] for j in range(N)])
        dw[1] = 2 / N * sum([(pred[j] - Y[j]) * X[j][1] for j in range(N)])
        db = 2 / N * sum([(pred[j] - Y[j]) for j in range(N)])
        self.w[0] -= dw[0] * MU
        self.w[1] -= dw[1] * MU
        self.b -= db * MU
        return err


f = open('prices.txt')

X, Y = [], []
for line in f:
    inp = line.split(',')
    X.append([float(inp[0]), float(inp[1])])
    Y.append(float(inp[2]))

last_error = 239
cnt = 0
linreg = LinearRegressor()

while(cnt < 100):
    last_error = linreg.train(X, Y)
    cnt += 1
    print ('batch #' + str(cnt) + ', MSE = ' + str(last_error))

print('w1 = ' + str(linreg.w[0]))
print('w2 = ' + str(linreg.w[1]))
print('b = ' + str(linreg.b))

while(True):
    print('Enter area and number of rooms to get price')
    inp = input().split(' ')
    x1 = float(inp[0])
    x2 = float(inp[1])
    print(linreg.predict([[x1, x2]])[0])



