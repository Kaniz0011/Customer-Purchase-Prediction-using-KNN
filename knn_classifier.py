import operator
from collections import Counter
class KNNeighbor:
    def __init__(self,k):
        self.k=k

    def fit(self,x_train,y_train):
        self.x_train=x_train
        self.y_train=y_train
        print('trainnig done')

    def predict(self,x_test):
        #self.x_test=x_test
        distance={}
        Counter=1
        for i in self.x_train:
            distance[Counter]=((x_test[0][0]-i[0])**2 +(x_test[0][1]-i[1]) ** 2)**1/2
            Counter=Counter+1
        distance=sorted(distance.items(),key=operator.itemgetter(1),reverse=True)
        self.classify(distance=distance[:self.k])
    def classify(self,distance):
        label=[]
        for i in distance:
            label.append(self.y_train[i[0]])
        return Counter(label).most_common()[0][0]