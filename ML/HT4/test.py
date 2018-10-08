from knn import KNNClassifier
from svm import SVM
import random
import copy
from scipy.stats import wilcoxon
NUMBER_OF_FOLDS = 5

def chunkify(lst):
    return [lst[i::NUMBER_OF_FOLDS] for i in range(NUMBER_OF_FOLDS)]

print('===========================')
print('testing started')


f = open('chips.txt')
points = []

for line in f:
    inp = line.split(',')
    points.append([float(inp[0]), float(inp[1]), int(inp[2])])

random.shuffle(points)
n_points = len(points)
points = chunkify(points)

acc1, acc2 = [], []

for i in range(NUMBER_OF_FOLDS):
    train, test = [], []
    for j in range(NUMBER_OF_FOLDS):
        if (i != j):
            train.extend(points[j])
        else:
            test = points[j]
    knn = KNNClassifier()
    svm = SVM()
    train2 = copy.deepcopy(train)
    knn.fit(train)
    svm.fit_transform(train2)
    k_right, s_right = 0, 0
    for point in test:
        k_pred = knn.predict([point[0], point[1]])
        s_pred = svm.predict([point[0], point[1]])
        s_pred = (s_pred + 1) / 2
        if (k_pred == point[2]):
            k_right = k_right + 1
        if (s_pred == point[2]):
            s_right = s_right + 1
    acc1.append(float(k_right / len(test)))
    acc2.append(float(s_right / len(test)))


a = wilcoxon(acc1, acc2)
print(a)