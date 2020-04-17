import numpy as np

class Model:

    def __init__(self):
        self.w = None
        self.threshold = None
        self.thresholdY = None
        self.inv = False
        self.scale = None


    def within_class_variance(self, X):
        mean = np.mean(X, axis=1, keepdims=True)
        cv = X - mean
        cv = np.matmul(cv, cv.T)
        return cv


    def segregate_classes(self, data):
        '''
            Takes as input examples (with target) stacked in rows. Returns examples stacked as column vectors (without target).
        '''
        X0 = []
        X1 = []
        for i in data:
            if i[-1] == 1:
                X1.append(i[:-1])
            else:
                X0.append(i[:-1])
        X0 = np.array(X0)
        X1 = np.array(X1)
        return X0.T, X1.T


    def learn_discriminant(self, X0, X1):
        # estimate means of each class
        mean0 = np.mean(X0, axis=1, keepdims=True)
        mean1 = np.mean(X1, axis=1, keepdims=True)
        # find within class variance for each class
        cv0 = self.within_class_variance(X0)
        cv1 = self.within_class_variance(X1)
        # calculate w (the direction vector of the discriminant)
        Sw = cv0 + cv1
        self.w = np.matmul(np.linalg.inv(Sw), mean1 - mean0).reshape(-1)
        # scale w so that w[0] = 1
        self.scale = 1/self.w[0]
        self.w = self.w*self.scale

    def transform(self, data):
        return np.matmul(self.w.T, data)


    def discriminate(self, data):
        # segregate data according to classes
        X0, X1 = self.segregate_classes(data)
        # learn the discriminant using the data
        self.learn_discriminant(X0, X1)
        # transform the points
        tX0 = np.matmul(self.w.T, X0)
        tX1 = np.matmul(self.w.T, X1)
        self.fix_threshold(tX0, tX1)
        return tX0, tX1, self.threshold, self.thresholdY

    
    def fix_threshold(self, X0, X1):
        mu0 = np.mean(X0)
        s0 = np.std(X0)
        mu1 = np.mean(X1)
        s1 = np.std(X1)
        if mu0 > mu1:
            self.inv = True
            mu0, mu1 = mu1, mu0
            s0, s1 = s1, s0
            X0, X1 = X1, X0
        # use binary search to find intersection of the normal distributions
        l = min(np.amin(X0), np.amin(X1))
        r = min(np.amax(X0), np.amax(X1))
        intersection = None
        while np.abs(l - r) > 1e-5:
            mid = l + (r - l)/2
            p0 = np.sqrt(1/(2*np.pi))*(1/s0)*np.exp(-((mid - mu0)**2/(2*s0*s0)))
            p1 = np.sqrt(1/(2*np.pi))*(1/s1)*np.exp(-((mid - mu1)**2/(2*s1*s1)))
            if p0 >= p1:
                l = mid
                intersection = mid
            elif p0 < p1:
                r = mid
            else:
                intersection = mid
        self.threshold = intersection
        self.thresholdY = p0


    def predict(self, data):
        data = data.T
        preds = self.transform(data)
        preds[preds >= self.threshold] = 1
        preds[preds != 1] = 0
        if self.inv:
            preds = 1 - preds
        return preds


    def get_w(self):
        return self.w

    def get_scale(self):
        return self.scale