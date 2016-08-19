import scipy.io as sio
from scipy.spatial import cKDTree as KDTree
from collections import Counter
from math import exp,log
from random import sample
from sklearn.metrics import roc_curve,auc
import numpy as np
import pickle




class Semi_KNN:
    def __init__(self,unlabel_data,label_data,label_list):
        self.data = data
        self.unlabel_data = unlabel_data
        self.label_data = label_data
        self.label_list = label_list
        self.K = 3;
        self.kdtree = None
        self.entropy = None
        self.err = None  #error rate

        self.tpr = []  #true positive rate
        self.fpr = []  #False Positive rate
        self.thres = [] #thresholds in ROC
        self.AUC = [] #AUC of ROC


    #define training
    def train(self,k):
        #k for the propagation speed in iterations

        #iteratively apply Semi-KNN for label propagation
        while True:
            print('unlabel_data size:',self.unlabel_data.shape)
            #build the tree from labeled data
            self.kdtree = KDTree(self.label_data)

            if self.unlabel_data.size == 0:
                return self.kdtree

            # search the k-nearest unlabel points
            nodes = np.zeros((k,2+self.K))     #node[avg_dist,unlabel_id,label_id1,label_id2,label_id3]
            nodes = np.array(nodes)

            for i in range(self.unlabel_data.shape[0]):          # search all unlabeled examples
                node_dist,node_id = self.kdtree.query(self.unlabel_data[i],self.K)
                node_dist = np.mean(node_dist)
                node = [node_dist,i]
                node.extend(node_id)

                if i<k:
                    nodes[i,:] = node
                elif np.max(nodes[:,0])>node_dist:
                    id = np.argmax(nodes[:,0])
                    nodes[id,:]= node
            #labeled data expansion and unlabeled data reduction
            nshape = nodes.shape
            print(nshape)
            #reshape new data sets
            lshape = self.label_data.shape
            newshape_ldata = (nshape[0]+lshape[0],lshape[1])
            newshape_label = (nshape[0]+lshape[0],1)
            self.label_data = np.resize(self.label_data,newshape_ldata)
            self.label_list = np.resize(self.label_list, newshape_label)

            #updata labels and label_data
            for j in range(nodes.shape[0]):
                #label data propagation
                self.label_data[lshape[0]+j,:]=self.unlabel_data[nodes[j,1],:]

                #label_list propagation
                neighbor_labels = self.label_list[[int(x) for x in nodes[j,2:2+(self.K)]]].flatten()

                if neighbor_labels.shape != np.unique(neighbor_labels).shape:
                    c = Counter(neighbor_labels)
                    label = c.most_common(1)
                    label = label[0]
                    label = label[0]
                else:                           #if N-nearest neighbors share not labels, the first one is the closest one
                    label = neighbor_labels[0]


                self.label_list[lshape[0]+j] = label
            #updata unlabel_data
            for j in range(nodes.shape[0]):
                self.unlabel_data = np.delete(self.unlabel_data,nodes[j,1],0)

    #define test
    def test(self,test_data,labels):
        preds = []
        neighborss = []
        # prediction
        for i in range(test_data.shape[0]):
            dists,neighbors = self.kdtree.query(test_data[i,:],self.K)

            if neighbors.shape != np.unique(neighbors).shape:
                c = Counter(neighbors)
                label = c.most_common(1)
                label = label[0]
                label = label[0]
            else:                       # if N-nearest neighbors do not share labels, the first one is the closest one
                label = neighbors[0]

            neighborss.append(label)
            preds.append(self.label_list[label])

        #error rate and entropy calculation
        diff = labels.__sub__(preds)
        false = diff.nonzero()
        self.err = false[0].shape[0]/labels.shape[0]
        self.entropy = 1-1/(1+exp(log(sum(abs(labels.__sub__(preds))))))

        #ROC calculation
        uq_labels = np.unique(labels)

        for i in range(uq_labels.shape[0]):
            tpr,fpr,thres = roc_curve(labels,preds,uq_labels[i])
            self.tpr.append(tpr)
            self.fpr.append(fpr)
            self.thres.append(thres)
            self.AUC.append(auc(fpr,tpr,reorder=True))


#main function

#data loading
AUCB = pickle.load(open('AUC.txt','rb'))
ACCB = pickle.load(open('acc.txt','rb'))

data = sio.loadmat("data_300.mat")
data = data['s']
data = data[0, 0]
udata = data['udata']
labels = data['labels']
ldata = data['ldata']
acc = []
AUC = []

entropy = []
for j in range(5):
    data_id = range(0,ldata.shape[0])
    test_id = sample(data_id,int(ldata.shape[0]/5)) #randomely pick 1/5 labeled data as test data

    tedata = ldata.copy(test_id)
    telabels = labels.copy(test_id)

    trdata = np.delete(ldata,test_id,0)
    trlabels = np.delete(labels,test_id,0)


    model = Semi_KNN(udata,trdata,trlabels)
    model.train(100)
    model.test(tedata,telabels)
    entropy.append(model.entropy)
    acc.append(model.err)
    AUC.append(model.AUC)

pickle.dump(acc, open('acc.txt', 'wb'))
pickle.dump(AUC, open('AUC.txt', 'wb'))






