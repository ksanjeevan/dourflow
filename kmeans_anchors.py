
from net.netparams import YoloParams
from net.utils import compute_iou, parse_annotation
import numpy as np
from scipy.spatial.distance import cdist

# See https://arxiv.org/abs/1612.08242
NUM_CENTROIDS = 5


def weighted_choice(choices):

    r = np.random.uniform(0, np.sum(choices))
    upto = 0
    for c, w in enumerate(choices):
        if upto + w >= r:
            return c
        upto += w
    return 0

class KMeans:

    def __init__(self, k):

        self.k = k
        self.diff_thresh = 1
        self.distf = IoU_dist
        #self.distf = lambda x,y: (x[0]-y[0])**2 + (x[1]-y[1])**2

    def fit(self, data):
        initial_centroids = self.init_centroids_kpp(data)

        self.centroids, self.clusters = self.cluster_data(data, initial_centroids)
        return self.centroids, self.clusters
    
    def init_centroids_kpp(self, data):

        centroids = []
        
        random_index = np.random.randint(len(data))
        centroids.append(data[random_index])

        while len(centroids) < self.k:
            
            prob_array = np.apply_along_axis(lambda x:
                self.mindist2(x, centroids), 1, data)

            norm = sum(prob_array)
            prob_array /= (norm + 1e-8)
            
            new_index = weighted_choice(prob_array)
            centroids.append(data[new_index])

        return np.array(centroids)


    def mindist2(self, x, centroids):
        dists = np.apply_along_axis(lambda c: self.distf(x, c),1, centroids)
        return np.min(dists) * np.min(dists)


    def cluster_data(self, data, initial_centroids):
        centroids = initial_centroids
        clusters = []
        counter = 0
        while True:
            old_clusters = clusters 
            old_centroids = centroids

            clusters = self.clusterfy(data, centroids)

            centroids = self.recalc_centroids(data, clusters)

            # Kmeans stopping condition based on some centroid shift delta?
            if len(old_clusters)>0:
                num_diffs = np.sum(old_clusters != clusters)
                print("Iteration = %d, Delta = %d"%(counter, num_diffs), flush=True)
                
                if num_diffs <= self.diff_thresh:
                    break
            counter += 1

        return centroids, clusters

    def clusterfy(self, data, centroids):
        return np.apply_along_axis(lambda d:
            np.argmin(cdist([d], centroids, self.distf)[0]), 1, data)


    def recalc_centroids(self, data, clusters):
        
        new_centroids = []

        for centroid_index in range(self.k):        
            
            centroid_data_idxs = np.where(clusters==centroid_index)[0]
            centroid_data = data[centroid_data_idxs]
            new_centroids.append( np.mean(centroid_data, axis=0) )

        return np.array(new_centroids)


def IoU_dist(x, c):
    return 1. - compute_iou([0,0,x[0],x[1]], [0,0,c[0],c[1]])




def exrtract_wh(img):
    result = []
    pixel_height = img['height']
    pixel_width = img['width']

    fact_pixel_grid_h = YoloParams.GRID_SIZE / pixel_height
    fact_pixel_grid_w = YoloParams.GRID_SIZE / pixel_width

    for obj in img['object']:
        grid_w = (obj['xmax'] - obj['xmin']) *  fact_pixel_grid_w
        grid_h = (obj['ymax'] - obj['ymin']) *  fact_pixel_grid_h
        result.append( [grid_w, grid_h] )

    return result

def gen_anchors(fname):

    imgs = parse_annotation(YoloParams.TRAIN_ANN_PATH,YoloParams.TRAIN_IMG_PATH)

    data_wh = []
    for img in imgs:
        data_wh += exrtract_wh(img)

    clustering = KMeans(NUM_CENTROIDS)

    centroids, _ = clustering.fit(np.array(data_wh))
    anchors = list(centroids.flatten())

    anchors_text = "".join(["%.5f, "%a \
    if i < len(anchors)-1 else "%.5f"%a for i,a in enumerate(anchors)])
    
    fname = fname if fname != 'custom_' else 'custom_anchors.txt'
    
    with open(fname,'w') as f:
        f.write("%s"%anchors_text)

    print("\nAnchors: \n")
    print(anchors_text)
    print("\n\tSored at: %s\n"%(fname))

    return anchors


def test():
    import matplotlib.pyplot as plt

    data1 = np.random.multivariate_normal([0,0], [[5,0],[0,5]], size=1000)
    data2 = np.random.multivariate_normal([0,10], [[5,0],[0,3]], size=700)
    data3 = np.random.multivariate_normal([10,0], [[2,0],[0,5]], size=900)

    data = np.concatenate([data1, data2, data3], axis=0)

    clust = KMeans(3)
    centroids, clusters = clust.fit(data)
    colors = ['c', 'g', 'r']

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)

    for k in range(len(centroids)):
        clust_data = data[np.where(clusters==k)[0]]
        x,y = clust_data.T
        ax.scatter(x,y, color=colors[k])

    x,y = centroids.T
    ax.scatter(x,y, color='k')

    ax.set_title('Test')

    fig.savefig('test.png', format='png')
    plt.close()




if __name__ == '__main__':

    
    gen_anchors()
    