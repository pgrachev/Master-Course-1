import numpy as np
import random
import matplotlib.pyplot as mpl

from tensorflow.examples.tutorials.mnist import input_data
class HammingNN:
    samples = []
    w = []
    mean = 0.0
    vec_dim = None
    n_samples = None

    def to_binary(self, sample):
        return np.sign(sample - self.mean)

    def train(self, samples):
        self.mean = np.sum(samples) / (samples.shape[0] * samples.shape[1])
        self.vec_dim = len(samples[0])
        self.n_samples = len(samples)
        self.samples = self.to_binary(samples)
        self.w = np.zeros([self.vec_dim, self.vec_dim])
        print(self.samples[0])
        print(np.outer(self.samples[0], self.samples[0]))
        for d in range(self.n_samples):
            self.w += np.outer(self.samples[d], self.samples[d])
        print(self.w)
        self.w *= 1 / self.vec_dim
        for i in range(self.vec_dim):
            self.w[i][i] = 0.0
        print(self.w)

    def apply(self, data):
        result = np.zeros(self.vec_dim)
        for j in range(self.vec_dim):
            for i in range(self.vec_dim):
                result[j] += self.w[i][j] * data[i]
            result[j] = np.sign(result[j])
        return result

    def converge(self, sample):
        sample = self.to_binary(sample)
        s = np.array(sample)
        s = np.reshape(s, (28, 28))
        mpl.imshow(s)
        mpl.show()
        prev = sample
        next = self.apply(prev)
        a = (prev == next)
        while (not (prev == next).all()):
            prev = next
            next = self.apply(prev)
        p = np.array(prev)
        p = np.reshape(p, (28, 28))
        mpl.imshow(p)
        mpl.show()
        return prev

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

X_train = np.vstack([img.reshape(-1,) for img in mnist.train.images])
y_train = mnist.train.labels

random.shuffle(X_train)

model = HammingNN()
model.train(X_train[:100])
model.converge(X_train[101])
