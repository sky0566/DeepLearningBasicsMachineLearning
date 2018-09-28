from sklearn.feature_extraction import DictVectorizer  # sklearn只支持Integer数据，用来转化
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

# Read in the csv file and put features into list of dict and list of class label
allElectronicsData = open(r'D:\develop\PythonProject\DeepLearningBasicsMachineLearning\DecisionTree\AllElectronics.csv', 'rt') # Python2中是rb
reader = csv.reader(allElectronicsData)
headers = next(reader)  # 表头 Python2中是reader.next()

# print(headers)

featureList = []  # 特征值information
lableList = [] # 最后标签

for row in reader:
    lableList.append(row[len(row) - 1])
    rowDict = {}  # 对应字典
    for i in range(1, len(row) - 1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

print(featureList)

# Vectorize features
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()

print("dummyX:" + str(dummyX))
print(vec.get_feature_names())

print("labelList:" + str(lableList))

# Vectorize class labels
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(lableList)
print("dummY:" + str(dummyY))

# Using decision tree for classification
# clf tree.DesicionTreeClassifier()
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print("clf:" + str(clf))

# Visulize model
with open("allElectronicInformationGainOri.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)  # feature_names 是返回以前的特征值

oneRowX = dummyX[0, :]
print("oneRowX: " + str(oneRowX))

newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX: " + str(newRowX))

predictedY = clf.predict(newRowX.reshape(1, -1)) #需要将数据reshape(1, -1)处理
print("predictedY: " + str(predictedY))
