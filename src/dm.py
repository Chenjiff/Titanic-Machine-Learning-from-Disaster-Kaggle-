
import pandas as pd
import numpy as np
import lightgbm as lgb
import gc




pd.set_option('display.max_columns',None)



#读取数据
train = pd.read_csv("train.csv", names=['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'], 
    dtype={'Survived': np.int64, 'Pclass': np.int64, 'Age': np.float64, 'SibSp': np.int8, 'Parch': np.int8, 'Fare': np.float64})
test = pd.read_csv("test.csv", names=['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'], 
    dtype={'PassengerId': np.int64, 'Pclass': np.int64, 'Age': np.float64, 'SibSp': np.int8, 'Parch': np.int8, 'Fare': np.float64})



#将表中的字符值编码成可以训练的数值类型
class2id = {}
id2class = {}
def processStr(baseTable):
    resTable = baseTable
    cat_columns = ['Name','Sex','Cabin','Embarked']
    for c in cat_columns:
        resTable[c] = resTable[c].apply(lambda x: x if type(x)==str else str(x))
        sort_temp = sorted(list(set(resTable[c])))
        class2id[c+'2id'] = dict(zip(sort_temp, range(1, len(sort_temp)+1)))
        resTable[c] = resTable[c].apply(lambda x: class2id[c+'2id'][x])
    return resTable

def processNan(baseTable):
    baseTable['Age'] = baseTable['Age'].fillna(baseTable['Age'].mean())
    baseTable['Embarked'] = baseTable['Embarked'].fillna(baseTable['Embarked'].mode()[0])
    baseTable['Fare'] = baseTable['Fare'].fillna(baseTable['Fare'].mean())
    return baseTable

def fea_process(baseTable):
    temp = []
    for i in list(baseTable["Ticket"]):
        if not i.isdigit() :
            temp.append(i.replace(".","").replace("/","").strip().split(' ')[0]) 
        else:
            temp.append("Null")
    baseTable["Ticket"] = temp
    baseTable = pd.get_dummies(baseTable, columns = ["Ticket"], prefix="T")
    baseTable['Fsize'] = baseTable['SibSp'] + baseTable['Parch'];
    baseTable['Havef'] = baseTable['Fsize'] != 0;
    baseTable['Fc'] = pd.qcut(baseTable['Fare'], 4, labels=[1, 2, 3, 4]);
    baseTable['Ac'] = pd.cut(baseTable['Age'], bins=[0, 12, 20, 50, 120], labels=[1,2,3,4]);
    return baseTable


train = processNan(train)
train = processStr(train)
train = fea_process(train)
test = processNan(test)
test = processStr(test)
test = fea_process(test)
train.to_csv("train_prod2.csv",index=0)
test.to_csv("test_prod2.csv",index=0)

#训练模型

from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.preprocessing import Imputer
from sklearn.ensemble import ExtraTreesClassifier

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

print("开始训练：")
param = {
        'learning_rate': 0.002,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'max_depth': 4,
        'objective': 'multiclass',
        'num_class': 7,
        'num_leaves': 15,
        'min_data_in_leaf': 10,
        'max_bin': 300,
        'metric': 'multi_error'
        }

X = train.drop(['PassengerId','Name','Survived'], axis=1)
y = train['Survived']
pid = test['PassengerId']
test = test.drop(['Name','PassengerId'], axis=1)
fea_name = list(X.columns)
cata=['Pclass','Sex','Cabin','Havef','Fc','Ac','Embarked'] 

xx_score = []
cv_pred = []
skf = StratifiedKFold(n_splits=3, random_state=1030, shuffle=True)
for index, (train_index, vali_index) in enumerate(skf.split(X, y)):
    print(index)
    x_train, y_train, x_vali, y_vali = np.array(X)[train_index], np.array(y)[train_index], np.array(X)[vali_index], np.array(y)[vali_index]
    train = lgb.Dataset(x_train, y_train)
    vali =lgb.Dataset(x_vali, y_vali)
    print("training start...")
    #model = lgb.train(param, train, feature_name=fea_name, categorical_feature=cata, num_boost_round=1000, valid_sets=[vali])
    model = lgb.train(param, train, feature_name=fea_name, categorical_feature=cata, num_boost_round=1000, valid_sets=[vali])
    xx_pred = model.predict(x_vali,num_iteration=model.best_iteration)
    xx_pred = [np.argmax(x) for x in xx_pred]
    xx_score.append(f1_score(y_vali,xx_pred,average='weighted'))
    y_test = model.predict(test,num_iteration=model.best_iteration)
    y_test = [np.argmax(x) for x in y_test]
    if index == 0:
        cv_pred = np.array(y_test).reshape(-1, 1)
    else:
        cv_pred = np.hstack((cv_pred, np.array(y_test).reshape(-1, 1)))
        
submit = []
for line in cv_pred:
    submit.append(np.argmax(np.bincount(line)))
baseTable = pd.DataFrame({'PassengerId':pid.as_matrix(),'Survived':submit})
baseTable.to_csv('gender_submission.csv',index=False)


