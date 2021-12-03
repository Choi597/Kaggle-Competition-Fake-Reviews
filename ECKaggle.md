

```python
import argparse
import os
import sys
import pickle
import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
#print(f'pandas version {pd.__version__}')
#print(f'Sklearn version {sklearn.__version__}')
```


```python
try:
    from sklearn.externals import joblib
except:
    import joblib
```


```python
test_file = None
train_file = None
validation_file = None
joblib_file = "LR_model.pkl"
    
#parser = argparse.ArgumentParser()
#group1 = parser.add_mutually_exclusive_group(required=True)
#group1.add_argument('-e', '--test', help='Test attributes (to predict)')
#group1.add_argument('-n', '--train', help='Train data')
#parser.add_argument('-v', '--validation', help='Validation data')

Train = False
Test = False
Validation = False
    
file_train = pd.read_csv('reviews_train.csv',quotechar='"',usecols=[0,1,2,3],dtype={'real review?': int,'category': str, 'rating': int, 'text_': str})
file_test = pd.read_csv('reviews_test_attributes.csv',quotechar='"',usecols=[0,1,2,3,4],dtype={'real review?': int,'category': str, 'rating': int, 'text_': str})
file_validation = pd.read_csv('reviews_validation.csv',quotechar='"',usecols=[0,1,2,3],dtype={'real review?': int,'category': str, 'rating': int, 'text_': str}) 

```


```python
#Tf-IDF on Training Data 
vectorizer = TfidfVectorizer(max_features=7500, ngram_range=(1,3))

corpora = file_train['text_'].astype(str).values.tolist()
vectorizer.fit(corpora)
X = vectorizer.transform(corpora)
print(X.shape)
```

    (37184, 7500)



```python
#TF-IDF on Validation Data
corporaVal = file_validation['text_'].astype(str).values.tolist()
X1 = vectorizer.transform(corporaVal)
print(X1.shape)
```

    (999, 7500)



```python
#vectorize Test Data
corporaTest = file_test['text_'].astype(str).values.tolist()
X2 = vectorizer.transform(corporaTest)
print(X2.shape)
```

    (2249, 7500)



```python
enc = preprocessing.OneHotEncoder()

#One-Hot Encoding on Category and Rating of Training Data
fittingC = enc.fit(file_train[['category']])
transformingC = fittingC.transform(file_train[['category']]).toarray()
fittingR = enc.fit(file_train[['rating']])
transformingR = fittingR.transform(file_train[['rating']]).toarray()
trainData = np.hstack((X.toarray(), transformingC, transformingR))
```


```python
#One-Hot Encoding on Category and Rating of Validation Data
fittingCVal = enc.fit(file_validation[['category']])
transformingCVal = fittingCVal.transform(file_validation[['category']]).toarray()
fittingRVal = enc.fit(file_validation[['rating']])
transformingRVal = fittingRVal.transform(file_validation[['rating']]).toarray()
valData = np.hstack((X1.toarray(), transformingCVal, transformingRVal))
valData
```




    array([[0., 0., 0., ..., 0., 0., 1.],
           [0., 0., 0., ..., 0., 1., 0.],
           [0., 0., 0., ..., 0., 0., 1.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 1., 0.],
           [0., 0., 0., ..., 0., 0., 1.]])




```python
fittingCTest = enc.fit(file_test[['category']])
transformingCTest = fittingCTest.transform(file_test[['category']]).toarray()
fittingRTest = enc.fit(file_test[['rating']])
transformingRTest = fittingRTest.transform(file_test[['rating']]).toarray()
testData = np.hstack((X2.toarray(), transformingCTest, transformingRTest))
testData
```




    array([[0., 0., 0., ..., 0., 0., 1.],
           [0., 0., 0., ..., 0., 1., 0.],
           [0., 0., 0., ..., 0., 0., 1.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 1., 0.],
           [0., 0., 0., ..., 0., 0., 1.]])




```python
best_accuracy = 0
for C in [10]:
    lr = LogisticRegression(penalty="l1", tol=0.001, C=C, fit_intercept=True, solver="saga", intercept_scaling=1, random_state=42)
    lr.fit(trainData, file_train['real review?'])
    
    # Get logistic regression predictions
    y_hat = lr.predict_proba(trainData)[:,1]
    validationy_hat = lr.predict_proba(valData)[:,1]
    accuracy = roc_auc_score(file_validation['real review?'], validationy_hat)
    
   
    

    #accuracy = (yval_pred == file_validation['real review?']).sum() / file_validation['real review?'].size
    print(f'Accuracy {accuracy}')
    #print(f'Fraction of non-zero model parameters {np.sum(lr.coef_==0)+1}')
    
    #if accuracy > best_accuracy:
        # Save logistic regression model
        #joblib.dump(lr, joblib_file)
        #best_accuracy = accuracy
        
    #currlist.append(accuracy)
#accList.append(currList)

        
```

    /apps/spack/scholar/fall20/apps/anaconda/2020.11-py38-gcc-4.8.5-djkvkvk/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn("The max_iter was reached which means "


    Accuracy 0.9842291547950592



```python
y_pred = lr.predict_proba(testData)[:,1]
predict = pd.DataFrame(columns = ["ID", "real review?"])
predict["ID"] = range(0, len(file_test))
predict["real review?"] = y_pred
predict.to_csv("predicted_labels.csv", index = False)

       
  
```


```python

```
