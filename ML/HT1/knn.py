import bisect
import random

NUMBER_OF_FOLDS = 5
metrics = ['euclid', 'minkwoski']
kernels = ['epanechnikov', 'triangular']
def distance(p1, p2, metric):
    ans = 0
    if(metric == 'minkowski'):
        for i in range(len(p1)):
            ans += abs(p1[i] - p2[i])
    else:
        for i in range(len(p1)):
            ans += (p1[i] - p2[i]) ** 2
        ans = ans ** (0.5)
    return ans

def importance(distance, kernel):
    if (kernel == 'triangular'):
        return max(0, 1.0 - abs(distance))
    if (kernel == 'epanechnikov'):
        return max(0, 3.0 / 4.0 * (1 - distance ** 2))

def chunkify(lst):
    return [lst[i::NUMBER_OF_FOLDS] for i in range(NUMBER_OF_FOLDS)]

class KNNClassifier:

    points = []
    flags = []
    k = None
    kernel = None
    metric = None

    def fit(self, points, k=5, kernel='epanechnikov', metric = 'euclid'):
        self.points, self.flas = [], []
        for point in points:
            self.points.append([point[0], point[1]])
            self.flags.append(point[2])
        self.k = k
        self.kernel = kernel
        self.metric = metric

    def findKNN(self, point):
        hierarchy, distances = [], []
        for i in range(len(self.points)):
            dist = distance(point, self.points[i], self.metric)
            position = bisect.bisect(distances, dist)
            hierarchy.insert(position, i)
            distances.insert(position, dist)
        return hierarchy[:self.k], distances[:self.k]

    def predict(self, point):
        neibs, dists = self.findKNN(point)
        dists = [dists[i] / dists[-1] for i in range(len(dists))]
        distr = [0.0, 0.0]
        for i in range(len(neibs)):
            distr[self.flags[neibs[i]]] += importance(dists[i], self.kernel)
        if(distr[0] > distr[1]):
            return 0
        return 1


classifier = KNNClassifier()

f = open('chips.txt')
points = []

for line in f:
    inp = line.split(',')
    points.append([float(inp[0]), float(inp[1]), int(inp[2])])

random.shuffle(points)
n_points = len(points)
points = chunkify(points)
maxf1 = 0
for kk in range(3, 20):
    print('number of neighbors: ' + str(kk))
    for kernel in kernels:
        print('|__kernel ' + str(kernel))
        for metric in metrics:
            print('|____metric ' + str(metric))
            meanf1 = 0
            T, F = [0, 0], [0, 0]
            for i in range(NUMBER_OF_FOLDS):
                train, test = [], []
                for j in range(NUMBER_OF_FOLDS):
                    if (i != j):
                        train.extend(points[j])
                    else:
                        test = points[j]
                classifier.fit(train, k=kk, kernel=kernel, metric=metric)
                for point in test:
                    pred = classifier.predict([point[0], point[1]])
                    if(pred == point[2]):
                        T[pred] = T[pred] + 1
                    else:
                        F[pred] = F[pred] + 1
            precision = T[1] / (T[1] + F[1])
            recall = T[1] / (T[1] + F[0])
            f1m = 2 * precision * recall / (precision + recall)
            maxf1 = max(maxf1, f1m)

            print ('F1 score: ' + str(float(f1m)) + '%.')

print('maxf1 = ' + str(maxf1))