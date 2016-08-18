import scipy.io as sio
from scipy.spatial import cKDTree as KDTree
import numpy as np

ldata = sio.loadmat("features.mat")
ldata = ldata['features']
ldata = ldata[0:200,:]
udata = sio.loadmat("features.mat")
udata = udata['features']
udata = udata[201:-1,:]
llist = [1,0]*100


def semiKNN(label_data,unlabel_data,label_list,k):
    #copy data from memory space
    #label_data = label_data.copy()
    #unlabel_data = unlabel_data.copy()

    #iteratively apply Semi-KNN
    while True:
        print('unlabel size:',unlabel_data.size)
        #build the tree from labeled data
        kdtree = KDTree(label_data)

        if unlabel_data.size == 0:
            return kdtree

        # search the k-nearest unlabel points
        nodes = np.zeros((k, 3))     #node[dist,unlabel_id,label_id]
        nodes = np.array(nodes)

        for i in range(unlabel_data.shape[0]):          # search all unlabeled examples
            node_dist,node_id = kdtree.query(unlabel_data[i])
            if i<k:
                nodes[i,:]=[node_dist,i,node_id]
            elif np.max(nodes[:,0])>node_dist:
                id = np.argmax(nodes[:,0])
                nodes[id,:]=[node_dist,i,node_id]

        #labeled data expansion and unlabeled data reduction
        nshape = nodes.shape
        lshape = label_data.shape
        newshape = (nshape[0]+lshape[0],lshape[1])
        label_data = np.resize(label_data,newshape)
        for j in range(nodes.shape[0]):
            label_data[lshape[0]+j]=unlabel_data[nodes[j,1]]
            label_list.append(label_list[int(nodes[j,2])])

        for j in range(nodes.shape[0]):
            unlabel_data = np.delete(unlabel_data,nodes[j,1],0)

#def evaluate(kdtree,test_data)



semiKNN(ldata,udata,llist,100)

a=1




