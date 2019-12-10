#%%
import numpy as np
import cv2
import os
from sklearn import preprocessing

def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def cosine_disntance(vec1, vec2):
    return np.dot(vec1, vec2)/ (np.linalg.norm(vec1) * np.linalg.norm(vec2))

class DenseDetector():
    def __init__(self, feature_scale=64, margin=0):
        # create a dense feature detector
        self.initFeatureScale = feature_scale
        self.XyStep = int(np.sqrt(feature_scale))
        self.margin = margin

    def detect(self, img):
        keypoints = np.zeros((1, self.initFeatureScale))
        rows, cols = img.shape
        for x in range(0, rows- 2*self.margin, self.margin):
            for y in range(0, cols - 2*self.margin, self.margin):
                keypoints = np.vstack((keypoints,np.reshape(img[x:x+self.XyStep, y:y+self.XyStep],(1,self.initFeatureScale))))
        return keypoints[1:,:]

def gen_sift_features(gray, step_size):
    lower_gray = cv2.pyrDown(gray)      #金字塔  
    dense = cv2.xfeatures2d.SIFT_create(103)
    kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, lower_gray.shape[0], step_size) for x in range(0, lower_gray.shape[1], step_size)]
    kp, desc = dense.compute(lower_gray, kp)
    return kp, desc

def get_dense_sampling_images(path, dim=8):
    # Number of each class (each class contains 100 pictures)
    N = 100

    # Classes dictionary
    Classes = ['bedroom', 'Coast', 'Forest', 'Highway', 'industrial', 'Insidecity', 'kitchen',
               'livingroom', 'Mountain', 'Office', 'OpenCountry', 'store', 'Street', 'Suburb', 'TallBuilding']
    dict = {'bedroom': 1, 'Coast': 2, 'Forest': 3, 'Highway': 4, 'industrial': 5, 'Insidecity': 6,
            'kitchen': 7, 'livingroom': 8, 'Mountain': 9, 'Office': 10, 'OpenCountry': 11, 'store': 12,
            'Street': 13, 'Suburb': 14, 'TallBuilding': 15}

    # set dimension of each picture
    d = dim ** 2

    # Initialization
    dense_sampling_image_total = np.zeros((1,128))
    dense_sampling_images = []
    #
    label = []
    for Class in Classes:
        for i in range(N):
            img = cv2.imread(os.path.join(path,'training', Class, str(i)+'.jpg'), cv2.IMREAD_GRAYSCALE)
            kp, desc = gen_sift_features(img, 4)                                  #每张图片的数组 e.g. 2000*128
            dense_sampling_image_total = np.vstack((dense_sampling_image_total,desc))
            dense_sampling_images.append(desc)
            label.append(dict[Class])

    return dense_sampling_image_total[1:,:], dense_sampling_images, label



def show_histogram(histogram):
    memo = []
    num = []
    for label in histogram:
        if label not in memo:
            memo.append(label)
            num.append(np.sum(histogram == label))
        else:
            continue
    # import matplotlib.pyplot as plt
    # plt.hist(x=num,bins=len(num))
    # plt.show()
    return np.array(num), np.array(memo)



#%%
path = os.path.abspath('.')
dense_sampling_image_total, dense_sampling_images, label = get_dense_sampling_images(path)
np.save('dense_sampling_image_total2.npy', dense_sampling_image_total)
np.save('dense_sampling_images2.npy', dense_sampling_images)
np.save('label2.npy', label)
print(1)
#%%
import numpy as np
dense_sampling_images = np.load('dense_sampling_images2.npy', allow_pickle=True)
label = np.load('label2.npy')


#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dense_sampling_images, label, test_size=.3)


#%%
print(X_train[0].shape)
X = X_train[0]
for i in range(1,len(X_train)):
    X = np.vstack((X,X_train[i]))
print(X.shape)

#%%
from sklearn.cluster import MiniBatchKMeans,KMeans

# cluster = MiniBatchKMeans(n_clusters=500, batch_size=50)
cluster = KMeans(n_clusters=500,random_state= 54, max_iter= 3000,n_jobs = 2,n_init = 2)
cluster.fit(X)
from sklearn.externals import joblib
joblib.dump(cluster, 'kmeans_model3.m')

#%%
km_model = joblib.load('kmeans_model3.m')

#%%
from sklearn import preprocessing
def get_image_presentation(dense_sampling_images, centroids_model):
    image_presentation = np.zeros((len(dense_sampling_images), len(centroids_model.cluster_centers_)))
    histograms = []
    count = 0
    for image in dense_sampling_images:
        hist = centroids_model.predict(image)
        histograms.append(hist)
        for label in hist:
            image_presentation[count][label] += 1
        image_presentation[count] /= len(image)
        count += 1

    return image_presentation, histograms

image_presentation_train, histograms_train = get_image_presentation(X_train, km_model)
image_presentation_test, histograms_test = get_image_presentation(X_test, km_model)
# np.save('image_presentation3.npy', image_presentation)
std=preprocessing.StandardScaler()
image_presentation_train=std.fit_transform(image_presentation_train)
image_presentation_test=std.fit_transform(image_presentation_test)
#%%
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# clf = OneVsRestClassifier(LogisticRegression()).fit(image_presentation_train, y_train)
clf = OneVsRestClassifier(SVC()).fit(image_presentation_train, y_train)

#%%
clf.score(image_presentation_test, y_test)


#%%
print(X_train.shape)
print(clf.score(image_presentation_test, y_test))

