import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


data = pd.read_csv('data\\dataTrain.csv')
data.drop(columns=['博物馆名字'], axis=1, inplace=True)
data = data.fillna(0)
y = data['是否'].apply(lambda x: 1 if x == '是' else 0).values
data = data.drop(columns=['是否'], axis=1)
print(data)
X = None
for c in data.columns:
    if str(data[c].dtype) == 'object':
        enc = preprocessing.OneHotEncoder()
        m = enc.fit_transform(data[c].values.reshape(-1, 1)).todense()
        arr_t = np.array(m)
    else:
        sca = StandardScaler()
        arr_t = sca.fit_transform(data[c].values.reshape(-1, 1))

    if X is None:
        X = arr_t
    else:
        X = np.concatenate([X, arr_t], axis=1)

print(X.shape)
# 建立SVM模型
svm = SVC().fit(X, y)
gnb = GaussianNB().fit(X, y)
clf_tree = tree.DecisionTreeClassifier().fit(X, y)
clf_lr = LogisticRegression().fit(X, y)

svm_pred = svm.predict(X)
gnb_pred = gnb.predict(X)
clf_tree_pred = clf_tree.predict(X)
clf_lr_pred = clf_lr.predict(X)

pred = np.concatenate([svm_pred, gnb_pred, clf_tree_pred, clf_lr_pred])
pred = pred.reshape((-1, 4))
print(pred.shape)

clf_rf = RandomForestClassifier(n_estimators=8).fit(pred, y)
cancer_target_pred = clf_rf.predict(pred)

# 预测训练集结果
print("预测前20个结果为：", cancer_target_pred[:20])
print('svm_pred：',accuracy_score(y,svm_pred))
print('gnb_pred：',accuracy_score(y,gnb_pred))
print('clf_tree_pred：',accuracy_score(y,gnb_pred))
print('clf_lr_pred：',accuracy_score(y,clf_lr_pred))
print('使用stacking预测数据的准确率为：',accuracy_score(y,cancer_target_pred))

