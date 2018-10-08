import numpy
import sklearn.decomposition as dec
import sklearn.linear_model as mod
import  sklearn.ensemble as en
import copy

TEST_SET_SIZE = 20
NUMBER_OF_COMPONENTS = 20

X = numpy.load('gist_vehicles_short/gist_vehicleimg.npy')
Y = numpy.load('gist_vehicles_short/gist_vehicletrgt.npy')
pca = dec.PCA(n_components=NUMBER_OF_COMPONENTS)
lsa = dec.TruncatedSVD(n_components=NUMBER_OF_COMPONENTS)

scd = mod.SGDClassifier()
rf = en.RandomForestClassifier(n_estimators=50)

reduction_methods = [None, pca, lsa]
reduction_method_names = ['None', 'pca', 'lsa']
classifiers = [scd, rf]
classifier_names = ['scd', 'rf']

trainX, trainY = X[:-TEST_SET_SIZE], Y[:-TEST_SET_SIZE]
testX, testY = X[-TEST_SET_SIZE:], Y[-TEST_SET_SIZE:]

for i in range(len(reduction_methods)):
    print('Reduction method: ' + reduction_method_names[i])
    if(reduction_method_names[i] == 'None'):
        currX = copy.deepcopy(trainX)
    else:
        currX = reduction_methods[i].fit_transform(X=trainX)
    for j in range(len(classifiers)):
        print('|__Classifier: ' + classifier_names[j])
        classifiers[j].fit(currX, trainY)
        if (reduction_method_names[i] == 'None'):
            currX2 = copy.deepcopy(testX)
        else:
            currX2 = reduction_methods[i].fit_transform(X=testX)
        print(len(currX2[0]))
        pred = classifiers[j].predict(currX2)
        tr = 0
        for it in range(len(testY)):
            if(pred[it] == testY[it]):
                tr = tr + 1
        print('|__Accuracy = ' + str(float(tr / TEST_SET_SIZE)))