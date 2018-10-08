
def first_number(x):
    return x[0]

def cht(index):
    return int((index + 1) / 2)

class CardiForest: #LSA requires O(10^8) of memory
    total_features = None
    key_features = None
    margins = []
    badness_index = []
    is_using = []

    def fit_transform(self, X, Y, key_feats = 10):
        n_pos, n_neg = 0, 0
        n_points = len(X)
        self.total_features = len(X[0])
        self.key_features = key_feats
        for y in Y:
            if(y > 0):
                n_pos = n_pos + 1
            else:
                n_neg = n_neg + 1

        for feat in range(len(X[0])):
            arr = []
            for ind in range(len(X)):
                arr.append([X[ind][feat], Y[ind]])
            arr.sort(key=first_number)


            self.badness_index.append(9999)
            self.margins.append([None, 0])
            pos_back, neg_back = 0, 0
            step = 0
            while (step < n_points - 1):
                if(arr[step][1] == 1):
                    pos_back = pos_back + 1
                else:
                    neg_back = neg_back + 1
                step = step + 1
                while(not step == n_points - 1 and arr[step][0] == arr[step - 1][0]):
                    if(arr[step][1] == 1):
                        pos_back = pos_back + 1
                    else:
                        neg_back = neg_back + 1
                    step = step + 1
                err1 = (n_neg - neg_back) + pos_back
                err2 = (n_pos - pos_back) + neg_back
                if (err1 < self.badness_index[-1]):
                    self.badness_index[-1] = err1
                    self.margins[-1] = [(arr[step][0] + arr[step - 1][0]) / 2.0, 1]

                if (err2 < self.badness_index[-1]):
                    self.badness_index[-1] = err2
                    self.margins[-1] = [(arr[step][0] + arr[step - 1][0]) / 2.0, -1]

        ar = []
        for i in range(self.total_features):
            ar.append([self.badness_index[i], i])
        ar.sort(key=first_number)
        for i in range(self.key_features):
            self.is_using.append(ar[i][1])

    def get_indicies(self):
        return self.is_using

    def predict(self, X):
        res = [0, 0]
        for i in self.is_using:
            if(self.margins[i][0] < X[i]):
                res[cht(self.margins[i][1])] = res[cht(self.margins[i][1])] + 1
            else:
                res[cht(-1 * self.margins[i][1])] = res[cht(-1 * self.margins[i][1])] + 1
        if(res[0] > res[1]):
            return -1
        else:
            return 1
'''

f = open('arcene_train.data', 'r')
X = []
for line in f:
    X0 = []
    features = line.split(" ")
    for feat in features:
        if(feat.isdigit()):
            X0.append(int(feat))
    X.append(X0)

f1 = open('arcene_train.labels', 'r')
Y = []
for line in f1:
    Y.append(int(line))

CF = CardiForest()
CF.fit_transform(X, Y)

f = open('arcene_valid.data', 'r')
f1 = open('arcene_valid.labels', 'r')

X = []
for line in f:
    X0 = []
    features = line.split(" ")
    for feat in features:
        if feat.isdigit():
            X0.append(int(feat))
    X.append(X0)

Y = []
for line in f1:
    Y.append(int(line))


right = 0
wrong = 0
for i in range(len(X)):
    if(CF.predict(X[i]) == Y[i]):
        right = right + 1
    else:
        wrong = wrong + 1

print(right / (right + wrong))

'''