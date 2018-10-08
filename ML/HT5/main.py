from sklearn.feature_selection import chi2
from sklearn.svm import SVC
from main3 import CardiForest

def get_exp(a):
    res = 0.0
    n = len(a)
    for elem in a:
        res += elem
    return float(res / n)

def get_var(a):
    res = 0.0
    n = len(a)
    for elem in a:
        res += elem * elem
    res = float(res / n)
    return res - get_exp(a) ** 2

def get_covar(a, b):
    res = 0.0
    n = len(a)
    for i in range(n):
        res += a[i] * b[i]
    res = float(res / n)
    return res - get_exp(a) - get_exp(b)

def get_correl(a, b):
    gva = get_var(a)
    gvb = get_var(b)
    if (gva > 0.0001 and gvb > 0.0001):
        return get_covar(a, b) / (get_var(a) * get_var(b))
    else: return 0.0

def SelectKBest(data, labels, k=5, metrics = 'correlation'):
    goodness = []
    if(metrics == 'chi2'):
        goodness = chi2(data, labels)[0]
    else:
        features = [[data[i][j] for i in range(len(data))] for j in range(len(data[0]))]
        goodness = [get_correl(features[i], labels) for i in range(len(features))]
    goodness_to_sort = [[goodness[i], i] for i in range(len(goodness))]
    goodness_to_sort.sort()
    ans = []
    for j in range(k):
        ans.append(goodness_to_sort[j][-1])
    return ans

def reduction_data(data, indecies):
    new_data = []
    for row in data:
        new_row = []
        for ind in indecies:
            new_row.append(row[ind])
        new_data.append(new_row)
    return new_data

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
CF.fit_transform(X, Y, key_feats=5)

random_indicies = [4307, 7010, 2029, 7273, 1282]

data1 = reduction_data(X, SelectKBest(X, Y, k=5, metrics='correlation'))
data2 = reduction_data(X, SelectKBest(X, Y, k=5, metrics='chi2'))
data3 = reduction_data(X, CF.get_indicies())
data4 = reduction_data(X, random_indicies)

s1 = SVC()
s1.fit(data1, Y)
s2 = SVC()
s2.fit(data2, Y)
s3 = SVC()
s3.fit(data3, Y)
s4 = SVC()
s4.fit(data4, Y)


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


data1 = reduction_data(X, SelectKBest(X, Y, k=5, metrics='correlation'))
data2 = reduction_data(X, SelectKBest(X, Y, k=5, metrics='chi2'))
data3 = reduction_data(X, CF.get_indicies())
data3 = reduction_data(X, random_indicies)
print(data1)
print(data2)
print(data3)
print(data4)

print(s1.score(data1, Y))
print(s2.score(data2, Y))
print(s3.score(data3, Y))
print(s4.score(data4, Y))