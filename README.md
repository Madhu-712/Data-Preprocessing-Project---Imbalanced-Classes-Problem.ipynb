# Data-Preprocessing-Project---Imbalanced-Classes-Problem.ipynb

Data Preprocessing Project - Imbalanced Classes Problem
Imbalanced classes is one of the major problems in machine learning. In this data preprocessing project, I discuss the imbalanced classes problem. I present Python implementation to deal with this problem.

Table of Contents
I have divided this project into various sections which are listed below:-

Introduction to imbalanced classes problem

Problems with imbalanced learning

Example of imbalanced classes

Approaches to handle imbalanced classes

Python implementation to illustrate class imbalance problem

Precision - Recall Curve

Random over-sampling the minority class

Random under-sampling the majority class

Apply tree-based algorithms

Random under-sampling and over-sampling with imbalanced-learn

Under-sampling : Tomek links

Under-sampling : Cluster Centroids

Over-sampling : SMOTE

Conclusion

1. Introduction to imbalanced classes problem
Any real world dataset may come along with several problems. The problem of imbalanced class is one of them. The problem of imbalanced classes arises when one set of classes dominate over another set of classes. The former is called majority class while the latter is called minority class. It causes the machine learning model to be more biased towards majority class. It causes poor classification of minority classes. Hence, this problem throw the question of “accuracy” out of question. This is a very common problem in machine learning where we have datasets with a disproportionate ratio of observations in each class.

Imbalanced classes problem is one of the major problems in the field of data science and machine learning. It is very important that we should properly deal with this problem and develop our machine learning model accordingly. If this not done, then we may end up with higher accuracy. But this higher accuracy is meaningless because it comes from a meaningless metric which is not suitable for the dataset in question. Hence, this higher accuracy no longer reliably measures model performance.

2. Problems with imbalanced learning
The problem of imbalanced classes is very common and it is bound to happen. For example, in the above example the number of patients who do not have the rare disease is much larger than the number of patients who have the rare disease. So, the model does not correctly classify the patients who have the rare disease. This is where the problem arises.

The problem of learning from imbalanced data have new and modern approaches. This learning from imbalanced data is referred to as imbalanced learning.

Significant problems may arise with imbalanced learning. These are as follows:-

The class distribution is skewed when the dataset has underrepresented data.

The high level of accuracy is simply misleading. In the previous example, it is high because most patients do not have the disease not because of the good model.

There may be inherent complex characteristics in the dataset. Imbalanced learning from such dataset requires new approaches, principles, tools and techniques. But, it cannot guarantee an efficient solution to the business problem.

3. Example of imbalanced classes
The problem of imbalanced classes may appear in many areas including the following:-

Disease detection

Fraud detection

Spam filtering

Earthquake prediction

4. Approaches to handle imbalanced classes
In this section, I will list various approaches to deal with the imbalanced class problem. These approaches may fall under two categories – dataset level approach and algorithmic ensemble techniques approach. The various methods to deal with imbalanced class problem are listed below. I will describe these techniques in more detail in the following sections.

Random Undersampling methods

Random Oversampling methods

Tree-based algorithms

Resampling with imbalanced-learn

Under-sampling : Tomek links

Under-sampling : Cluster Centroids

Over-sampling : SMOTE

I have discussed these methods in detail in the readme document.

5. Python implementation to illustrate class imbalance problem
Now, I will perform Python implementation to illustrate class imbalance problem.

Import Python libraries
I will start off by importing the required Python libraries.

# import Python libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
Import dataset
Now, I will import the dataset with the usual Python read_csv() function.

data = 'C:/datasets/creditcard.csv'

df = pd.read_csv(data)
Dataset description
I have used the Credit Card Fraud Detecttion dataset for this project. I have downloaded this project from the Kaggle website. This dataset can be found at the following url-

https://www.kaggle.com/mlg-ulb/creditcardfraud

This dataset contains transactions made by european credit card holders in September 2013. It represents transactions that occurred in two days. We have 492 fraudulent transactions out of total 284,807 transactions. This dataset is highly unbalanced, the positive class (frauds) account for only 0.172% of all transactions.

Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise. So, our target variable is Class variable.

Given the class imbalance ratio, it is recommended to measure the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.

Exploratory data analysis
Now, I will conduct exploratory data analysis to gain an insight into the dataset.

# check shape of dataset

df.shape
(284807, 31)
We can see that there are 284,807 instances and 31 columns in the dataset.

# preview of the dataset

df.head()
Time	V1	V2	V3	V4	V5	V6	V7	V8	V9	...	V21	V22	V23	V24	V25	V26	V27	V28	Amount	Class
0	0.0	-1.359807	-0.072781	2.536347	1.378155	-0.338321	0.462388	0.239599	0.098698	0.363787	...	-0.018307	0.277838	-0.110474	0.066928	0.128539	-0.189115	0.133558	-0.021053	149.62	0
1	0.0	1.191857	0.266151	0.166480	0.448154	0.060018	-0.082361	-0.078803	0.085102	-0.255425	...	-0.225775	-0.638672	0.101288	-0.339846	0.167170	0.125895	-0.008983	0.014724	2.69	0
2	1.0	-1.358354	-1.340163	1.773209	0.379780	-0.503198	1.800499	0.791461	0.247676	-1.514654	...	0.247998	0.771679	0.909412	-0.689281	-0.327642	-0.139097	-0.055353	-0.059752	378.66	0
3	1.0	-0.966272	-0.185226	1.792993	-0.863291	-0.010309	1.247203	0.237609	0.377436	-1.387024	...	-0.108300	0.005274	-0.190321	-1.175575	0.647376	-0.221929	0.062723	0.061458	123.50	0
4	2.0	-1.158233	0.877737	1.548718	0.403034	-0.407193	0.095921	0.592941	-0.270533	0.817739	...	-0.009431	0.798278	-0.137458	0.141267	-0.206010	0.502292	0.219422	0.215153	69.99	0
5 rows × 31 columns

The df.head() function gives the preview of the dataset. We can see that there is a Class column in the dataset which is our target variable.

I will check the distribution of the Class column with the value_counts() method as follows:-

# check the distribution of Class column

df['Class'].value_counts()
0    284315
1       492
Name: Class, dtype: int64
So, we have 492 fraudulent transactions out of total 284,807 transactions in the dataset. The Class column takes value 1 for 
fraudulent transactions and 0 for non-fraudulent transactions.

Now, I will find the percentage of labels 0 and 1 within the Class column.

# percentage of labels within the Class column

df['Class'].value_counts()/np.float(len(df))
0    0.998273
1    0.001727
Name: Class, dtype: float64
We can see that the Class column is highly imbalanced. It contains 99.82% labels as 0 and 0.17% labels as 1.

Now, I will plot the bar plot to confirm this.

# view the distribution of percentages within the Class column


(df['Class'].value_counts()/np.float(len(df))).plot.bar()
<matplotlib.axes._subplots.AxesSubplot at 0xb5820294a8>

The above bar plot confirms our finding that the Class variable is highly imbalanced.

Misleading accuracy for imbalanced classes
Now, I will demonstrate that accuracy is misleading for imbalanced classes. Most of the machine learning algorithms are designed to maximize the overall accuracy by default. But this maximum accuracy is misleading. We can confirm this with the following analysis.

I will fit a very simple Logistic Regression model using the default settings. I will train the classifier on the imbalanced dataset.

# declare feature vector and target variable

X = df.drop(['Class'], axis=1)
y = df['Class']
# import Logistic Regression classifier
from sklearn.linear_model import LogisticRegression


# instantiate the Logistic Regression classifier
logreg = LogisticRegression()


# fit the classifier to the imbalanced data
clf = logreg.fit(X, y)


# predict on the training data
y_pred = clf.predict(X)
Now, I have trained the model. I will check its accuracy.

# import the accuracy metric
from sklearn.metrics import accuracy_score


# print the accuracy
accuracy = accuracy_score(y_pred, y)

print("Accuracy : %.2f%%" % (accuracy * 100.0))
Accuracy : 99.90%
Accuracy paradox
Thus, our Logistic Regression model for credit card fraud detection has an accuracy of 99.90%. It means that for each 100 transactions it classified, 99.90% were classified as genuine.

It does not mean that our model performance is excellent. I have previously shown that our dataset have 99.90% genuine transactions and 0.1% fraudulent transactions. Our Logistic Regression classifier predicted all transactions as genuine. Then we have a accuracy of 99.90% because it correctly classified 99.90% transactions as genuine.

Thus, this algorithm is 99.90% accurate. But it was horrible at classifying fraudulent transactions. So, we should have other ways to measure the model performance. One such measure is confusion matrix described below.

Confusion matrix
A confusion matrix is a tool for summarizing the performance of a classification algorithm. A confusion matrix will give us a clear picture of classification model performance and the types of errors produced by the model. It gives us a summary of correct and incorrect predictions broken down by each category. The summary is represented in a tabular form.

Four types of outcomes are possible while evaluating a classification model performance. These four outcomes are described below:-

True Positives (TP) – True Positives occur when we predict an observation belongs to a certain class and the observation actually belongs to that class.

True Negatives (TN) – True Negatives occur when we predict an observation does not belong to a certain class and the observation actually does not belong to that class.

False Positives (FP) – False Positives occur when we predict an observation belongs to a certain class but the observation actually does not belong to that class. This type of error is called Type I error.

False Negatives (FN) – False Negatives occur when we predict an observation does not belong to a certain class but the observation actually belongs to that class. This is a very serious error and it is called Type II error.

These four outcomes are summarized in a confusion matrix given below.

# import the metric
from sklearn.metrics import confusion_matrix


# print the confusion matrix
cnf_matrix = confusion_matrix(y, y_pred)


print('Confusion matrix:\n', cnf_matrix)
                             
Confusion matrix:
 [[284240     75]
 [   203    289]]
Interpretation of confusion matrix
Now, I will interpret the confusion matrix.

Out of the total 284315 transactions which were predicted genuine, the classifier predicted correctly 284240 of them. It means that the classifer predicted 284240 transactions as genuine and they were actually genuine. Also, it predicted 75 transactions as genuine but it were fraudulent. So, we have 284240 True Positives(TP) and 75 False Positives(FP).
Out of the total 492 transactions which were not predicted as genuine, the classifier predicted correctly 289 of them. It means that the classifer did not predict 289 transactions as genuine and they were actually not genuine. SO, they were fraudulent. Also, it did not predict 203 transactions as genuine but they were genuine. So, we have 289 True Negatives(TN) and 203 False Negatives(FN).
So, out of all the 284807 transactions, the classifier correctly predicted 284529 of them. Thus, we will get the accuracy of (284240+289)/(284240+289+75+203) = 99.90%.
But this is not the true picture. The confusion matrix allows us to obtain a true picture of the performance of the algorithm. The algorithm tries to predict the fraudulent transactions out of the total transactions. It correctly predicted 289 transactions as fraudulent out of all the 284807 transactions. In this case the accuracy becomes (289/284807)=0.10%.
Moreover, we have 203+289=492 transactions as fraudulent. The algorithm is correctly classifying 289 of them as fraudulent while it fails to predict 203 transactions which were fraudulent. In this case the accuracy becomes (289/492)=58.74%.
So, we can conclude that the accuracy of 99.90% is misleading because we have imbalanced classes. We need more subtle way to evaluate the performance of the model.

There is another metric called Classification Report which helps to evaluate model performance.

Classification report
Classification report is another way to evaluate the classification model performance. It displays the precision, recall, f1 and support scores for the model. I have described these terms in later sections.

We can plot a classification report as follows:-

# import the metric
from sklearn.metrics import classification_report


# print classification report
print("Classification Report:\n\n", classification_report(y, y_pred))
Classification Report:

               precision    recall  f1-score   support

           0       1.00      1.00      1.00    284315
           1       0.79      0.59      0.68       492

   micro avg       1.00      1.00      1.00    284807
   macro avg       0.90      0.79      0.84    284807
weighted avg       1.00      1.00      1.00    284807

Precision
Precision can be defined as the percentage of correctly predicted positive outcomes out of all the predicted positive outcomes. It can be given as the ratio of true positives (TP) to the sum of true and false positives (TP + FP).

Mathematically, precision can be defined as the ratio of TP to (TP + FP).

So, precision is more concerned with the positive class than the negative class.

Recall
Recall can be defined as the percentage of correctly predicted positive outcomes out of all the actual positive outcomes. It can be given as the ratio of true positives (TP) to the sum of true positives and false negatives (TP + FN). Recall is also called Sensitivity.

Mathematically, recall can be given as the ratio of TP to (TP + FN).

f1-score
f1-score is the weighted harmonic mean of precision and recall. The best possible f1-score would be 1.0 and the worst would be 0.0. f1-score is the harmonic mean of precision and recall. So, f1-score is always lower than accuracy measures as they embed precision and recall into their computation. The weighted average of f1-score should be used to compare classifier models, not global accuracy.

Support
Support is the actual number of occurrences of the class in our dataset. It classifies 284315 transactions as genuine and 492 transactions as fraudulent.

ROC Curve
Another tool to measure the classification model performance visually is ROC Curve. ROC Curve stands for Receiver Operating Characteristic Curve.

The ROC Curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold levels.

True Positive Rate (TPR) is also called Recall. It is defined as the ratio of TP to (TP + FN).

False Positive Rate (FPR) is defined as the ratio of FP to (FP + TN).

The Receiver Operating Characteristic Area Under Curve (ROC AUC) is the area under the ROC curve. The higher it is, the better the model is.

In the ROC Curve, we will focus on the TPR (True Positive Rate) and FPR (False Positive Rate) of a single point. This will give us the general performance of the ROC curve which consists of the TPR and FPR at various probability thresholds.

Precision - Recall Curve
Another tool to measure the classification model performance is Precision-Recall Curve. It is a useful metric which is used to evaluate a classifier model performance when classes are very imbalanced such as in this case. This Precision-Recall Curve shows the trade off between precision and recall.

In a Precision-Recall Curve, we plot Precision against Recall.

Precision is defined as the ratio of TP to (TP + FP).

Recall is defined as the ratio of TP to (TP + FN).

The Precision Recall Area Under Curve (PR AUC) is the area under the PR curve. The higher it is, the better the model is.

Difference between ROC AUC and PR AUC
Precision-Recall does not account for True Negatives (TN) unlike ROC AUC (TN is not a component of either Precision or Recall).
In the cases of class imbalance problem, we have many more negatives than positives. The Precision-Recall curve much better illustrates the difference between algorithms in the class imbalance problem cases where there are lot more negative examples than the positive examples. In these cases of class imbalances, we should use Precision-Recall Curve (PR AUC), otherwise we should use ROC AUC.
So, we can conclude that we should use PR AUC for cases where the class imbalance problem occurs. Otherwise, we should use ROC AUC.

6. Precision - Recall Curve
In the previous section, we conclude that we should use Precision-Recall Area Under Curve for cases where the class imbalance problem exists. Otherwise, we should use ROC-AUC (Receiver Operating Characteristic Area Under Curve).

Now, I will compute the average precision score.

# compute and print average precision score

from sklearn.metrics import average_precision_score

average_precision = average_precision_score(y_pred, y)

print('Average precision-recall score : {0:0.2f}'.format(average_precision))
Average precision-recall score : 0.47
Precision-Recall Curve gives us the correct accuracy in this imbalanced dataset case. We can see that we have a very poor accuracy for the model.

Now, I will plot the precision-recall curve.

from sklearn.metrics import precision_recall_curve 

precision, recall, thresholds = precision_recall_curve(y_pred, y)

# create plot
plt.plot(precision, recall, label='Precision-recall curve')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision-recall curve')
plt.legend(loc="lower left")
<matplotlib.legend.Legend at 0xb5847599e8>

7. Random over-sampling the minority class
Over-sampling is the process of randomly duplicating observations from the minority class in order to achieve a balanced dataset. So, it replicates the observations from minority class to balance the data. It is also known as upsampling. It may result in overfitting due to duplication of data points.

The most common way of over-sampling is to resample with replacement. I will proceed as follows:-

First, I will import the resampling module from Scikit-Learn.

# import resample module 

from sklearn.utils import resample
Now, I will create a new dataframe with an oversampled minority class as follows:-

At first, I will separate observations from Class variable into different DataFrames.
Now, I will resample the minority class with replacement. I will set the number of samples of minority class to match that of the majority class.
Finally, I will combine the oversampled minority class DataFrame with the original majority class DataFrame.
# separate the minority and majority classes
df_majority = df[df['Class']==0]
df_minority = df[df['Class']==1]
# oversample minority class

df_minority_oversampled = resample(df_minority, replace=True, n_samples=284315, random_state=0)      
# combine majority class with oversampled minority class

df_oversampled = pd.concat([df_majority, df_minority_oversampled])
# display new class value counts

df_oversampled['Class'].value_counts()
1    284315
0    284315
Name: Class, dtype: int64
Now, we can see that we have a balanced dataset. The ratio of the two class labels is now 1:1.

Now, I will plot the bar plot of the above two classes.

# view the distribution of percentages within the Class column


(df_oversampled['Class'].value_counts()/np.float(len(df_oversampled))).plot.bar()
<matplotlib.axes._subplots.AxesSubplot at 0xb58476ee48>

The above bar plot shows that we have a balanced dataset.

Now, I will train another model using Logistic Regression and check its accuracy, but this time on the balanced dataset.

# declare feature vector and target variable
X1 = df_oversampled.drop(['Class'], axis=1)
y1 = df_oversampled['Class']


# instantiate the Logistic Regression classifier
logreg1 = LogisticRegression()


# fit the classifier to the imbalanced data
clf1 = logreg1.fit(X1, y1)


# predict on the training data
y1_pred = clf1.predict(X1)


# print the accuracy
accuracy1 = accuracy_score(y1_pred, y1)

print("Accuracy : %.2f%%" % (accuracy1 * 100.0))
Accuracy : 93.76%
We now have a balanced dataset. Although the accuracy is slightly decreased, but it is still quite high and acceptable. This accuracy is more meaningful as a performance metric.

8. Random under-sampling the majority class
The under-sampling methods work with the majority class. In these methods, we randomly eliminate instances of the majority class. It reduces the number of observations from majority class to make the dataset balanced. This method is applicable when the dataset is huge and reducing the number of training samples make the dataset balanced.

The most common technique for under-sampling is resampling without replacement.

I will proceed exactly as in the case of random over-sampling.

# separate the minority and majority classes
df_majority = df[df['Class']==0]
df_minority = df[df['Class']==1]
# undersample majority class

df_majority_undersampled = resample(df_majority, replace=True, n_samples=492, random_state=0) 
# combine majority class with oversampled minority class

df_undersampled = pd.concat([df_minority, df_majority_undersampled])
# display new class value counts

df_undersampled['Class'].value_counts()
1    492
0    492
Name: Class, dtype: int64
Now, we can see that the new dataframe df_undersampled has fewer observations than the original one df and the ratio of the two classes is now 1:1.

Again, I will train a model using Logistic Regression classifier.

# declare feature vector and target variable
X2 = df_undersampled.drop(['Class'], axis=1)
y2 = df_undersampled['Class']


# instantiate the Logistic Regression classifier
logreg2 = LogisticRegression()


# fit the classifier to the imbalanced data
clf2 = logreg2.fit(X2, y2)


# predict on the training data
y2_pred = clf2.predict(X2)


# print the accuracy
accuracy2 = accuracy_score(y2_pred, y2)

print("Accuracy : %.2f%%" % (accuracy2 * 100.0))
Accuracy : 93.90%
Again, we can see that we have a slightly decreased accuracy but it is more meaningful now.

9. Apply Tree-Based Algorithms
# declare input features (X) and target variable (y)
X4 = df.drop('Class', axis=1)
y4 = df['Class']
# import Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
# instantiate the classifier 
clf4 = RandomForestClassifier()
# fit the classifier to the training data
clf4.fit(X4, y4)
 
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
# predict on training set
y4_pred = clf4.predict(X4)
# compute and print accuracy
accuracy4 = accuracy_score(y4_pred, y4)
print("Accuracy : %.2f%%" % (accuracy4 * 100.0))
Accuracy : 99.99%
# compute and print ROC-AUC

from sklearn.metrics import roc_auc_score

y4_prob = clf4.predict_proba(X4)
y4_prob = [p[1] for p in y4_prob]
print("ROC-AUC : " , roc_auc_score(y4, y4_prob))
ROC-AUC :  0.9999992243516689
10. Random under-sampling and over-sampling with imbalanced-learn
There is a Python library which enable us to handle the imbalanced datasets. It is called Imbalanced-Learn. It is a Python library which contains various algorithms to handle the imbalanced datasets. It can be easily installed with the pip command. This library contains a make_imbalance method to exasperate the level of class imbalance within a given dataset.

Now, I will demonstrate the technique of random undersampling and oversampling with imbalanced learn.

First of all, I will import the imbalanced learn library.

# import imbalanced learn library

import imblearn
Then, I will import the RandomUnderSampler class. It is a quick and easy way to balance the data by randomly selecting a subset of data for the targeted classes.

# import RandomUnderSampler
from imblearn.under_sampling import RandomUnderSampler

# instantiate the RandomUnderSampler
rus = RandomUnderSampler(return_indices=True)


# fit the RandomUnderSampler to the dataset
X_rus, y_rus, id_rus = rus.fit_sample(X, y)
# print the removed indices
print("Removed indices: ", id_rus)
Removed indices:  [194966 158207 239580  47508 164976   4059 244121   4712 129277 195768
 220169 159815 157385 226067 122412 186077 216500  36600 232308  61840
 216772 269873  49886 138921  64943 104600 211825 162198 236942 256405
 116084 165304 254299 217511  91142  67255   2349 132109 227416  75785
  23316 131322 177311  61790  91798 103220 103526  33083 148175 117300
 117437 251709 243617 136620  78369 177568 157649 150080 137441 277646
    988 213741 213602 264759 102266 166026 192696 269500 182970   7029
 138352 262530  99000 159383 225900 249330  14929 117795 252069  86625
 249970  58096 109913 195548  30897   8690  22107 261540 111780 105375
  62971 201607 177552  30981  84358 226572   7675  64315 172103 171021
  72979 208177  38876  63638 180868  76338 121268 264548 117134 182323
 254834 166395 235471 204943   9850 232780  83992  20930  54009 198064
 133443  81674 207050 274408 266475 165966 277813  23758  49157 222434
 178038   7746  72809 101544 198536 158586 102708 263512  63292 147707
  31677   6328 117195  33384  17731  59202  46117   2387  42641 195419
  99068 252335  81636  31663 136341  60555 120632 252749  77847  38097
 120550 197729  40557  45459 231116 131960 172075 190211 101223  15245
  29630  12298  41976 214451 248802 269876  19449  10430 123236  12237
 129734 144573 244632 270905 101667   6710   7139 234437  15658 246551
 230319 244236 257475  27540  67248 119614 270866 128596 146983   7481
 127789  89062  98494   7782 171395  13694 262656 240202  72635 265137
  35247 133304 253329 260581 204381 215806 209839  58584 280688 260875
 197410 244160  33589 192219  59500 178612 106610 272537 116951 227946
 241031 147854 235519 129102  54520 275960  37683  17697 136346 214118
 154999    661 209611 219382 227776  79642 114606 180877 269470 176024
 236924 228334 272616 241550  85666 138678 114154 116010 201161 202505
 143135  22461 269580 184552 216170 140159 239925 176351  97376 183893
 157472   8541  17800 162709 229779  94546  64270 269980 102331 179326
 181376 212424  19644 136198 193411  48301  42389 125628  96447  65219
 156953 238063 236407 118678 188004  88127 197928  83981 196266  58068
 173047 195404  78823 232805 200973 271144 141205 169895 226854 152573
 137340 249271 204025 101012 206959 184718 148995  19560  60100  93132
 228612  52144  94759 153881 202918 152068  87795 148397 253826 209436
  55565 156536 210650  52685 108761 130575 138829 278945  98843  30672
  74650  25075  37943  88289 232059  86024 124666 129281 168990 260644
  20605 259212 103199 105397 213227  69835  52662 161449   7192 234743
 161317 254693 131443 120272 272556  77072 224933 136289  65895 251918
 138989 156555 166011  69344 151419 228769 138165  72384 108491 249294
 242158 105805  31188  47452 107703 191593 108528 107788 195917 276293
 273099 166631 259123  52203  88536  23808 128556 234308  70648 266068
   1888 159394  60659  85037 272887 140314 152387 181888 153327  42232
  28043 162227 165238 124602 252553  59639  71697  88871  26193  73080
  36374 116525 171421 197736 193143 163500 235175 209073 236240 145689
  61701 169288 163571 134394 113957 225109 140344   1769 232132   2260
 184603 179724 185192   6070  18868 162478 209752  49614  91240 110724
  41575 112947  64470 109339 199665 251105 235051 263785 159364 241090
 273473  30352 232970 168645 234204  99853 116866 201284 144682 272207
 254527  47779 269575 161781 267340  45439 224250 181340   5375 173659
   5646  11315  74204  63810  91425 210100 171055 106752 179427 119743
  18362  30221    541    623   4920   6108   6329   6331   6334   6336
   6338   6427   6446   6472   6529   6609   6641   6717   6719   6734
   6774   6820   6870   6882   6899   6903   6971   8296   8312   8335
   8615   8617   8842   8845   8972   9035   9179   9252   9487   9509
  10204  10484  10497  10498  10568  10630  10690  10801  10891  10897
  11343  11710  11841  11880  12070  12108  12261  12369  14104  14170
  14197  14211  14338  15166  15204  15225  15451  15476  15506  15539
  15566  15736  15751  15781  15810  16415  16780  16863  17317  17366
  17407  17453  17480  18466  18472  18773  18809  20198  23308  23422
  26802  27362  27627  27738  27749  29687  30100  30314  30384  30398
  30442  30473  30496  31002  33276  39183  40085  40525  41395  41569
  41943  42007  42009  42473  42528  42549  42590  42609  42635  42674
  42696  42700  42741  42756  42769  42784  42856  42887  42936  42945
  42958  43061  43160  43204  43428  43624  43681  43773  44001  44091
  44223  44270  44556  45203  45732  46909  46918  46998  47802  48094
  50211  50537  52466  52521  52584  53591  53794  55401  56703  57248
  57470  57615  58422  58761  59539  61787  63421  63634  64329  64411
  64460  68067  68320  68522  68633  69498  69980  70141  70589  72757
  73784  73857  74496  74507  74794  75511  76555  76609  76929  77099
  77348  77387  77682  79525  79536  79835  79874  79883  80760  81186
  81609  82400  83053  83297  83417  84543  86155  87354  88258  88307
  88876  88897  89190  91671  92777  93424  93486  93788  94218  95534
  95597  96341  96789  96994  99506 100623 101509 102441 102442 102443
 102444 102445 102446 102782 105178 106679 106998 107067 107637 108258
 108708 111690 112840 114271 116139 116404 118308 119714 119781 120505
 120837 122479 123141 123201 123238 123270 123301 124036 124087 124115
 124176 125342 128479 131272 135718 137705 140786 141257 141258 141259
 141260 142405 142557 143188 143333 143334 143335 143336 143728 143731
 144104 144108 144754 145800 146790 147548 147605 149145 149357 149522
 149577 149587 149600 149869 149874 150601 150644 150647 150654 150660
 150661 150662 150663 150665 150666 150667 150668 150669 150677 150678
 150679 150680 150684 150687 150692 150697 150715 150925 151006 151007
 151008 151009 151011 151103 151196 151462 151519 151730 151807 152019
 152223 152295 153823 153835 153885 154234 154286 154371 154454 154587
 154633 154668 154670 154676 154684 154693 154694 154697 154718 154719
 154720 154960 156988 156990 157585 157868 157871 157918 163149 163586
 167184 167305 172787 176049 177195 178208 181966 182992 183106 184379
 189587 189701 189878 190368 191074 191267 191359 191544 191690 192382
 192529 192584 192687 195383 197586 198868 199896 201098 201601 203324
 203328 203700 204064 204079 204503 208651 212516 212644 213092 213116
 214662 214775 215132 215953 215984 218442 219025 219892 220725 221018
 221041 222133 222419 223366 223572 223578 223618 226814 226877 229712
 229730 230076 230476 231978 233258 234574 234632 234633 234705 235616
 235634 235644 237107 237426 238222 238366 238466 239499 239501 240222
 241254 241445 243393 243547 243699 243749 243848 244004 244333 245347
 245556 247673 247995 248296 248971 249167 249239 249607 249828 249963
 250761 251477 251866 251881 251891 251904 252124 252774 254344 254395
 255403 255556 258403 261056 261473 261925 262560 262826 263080 263274
 263324 263877 268375 272521 274382 274475 275992 276071 276864 279863
 280143 280149 281144 281674]
The above indices are removed from the original dataset.

Now, I will demonstrate random oversampling. The process will be the same as random undersampling.

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()

X_ros, y_ros = ros.fit_sample(X, y)
print(X_ros.shape[0] - X.shape[0], 'new random points generated')
283823 new random points generated
11. Under-sampling : Tomek links
Tomek links are defined as the two observations of different classes which are nearest neighbours of each other.

The figure below illustrate the concept of Tomek links-

Tomek%20links.jpg

We can see in the above image that the Tomek links (circled in green) are given by the pairs of red and blue data points that are nearest neighbors. Most of the classification algorithms face difficulty due to these points. So, I will remove these points and increase the separation gap between two classes. Now, the algorithms produce more reliable output.

This technique will not produce a balanced dataset. It will simply clean the dataset by removing the Tomek links. It may result in an easier classification problem. Thus, by removing the Tomek links, we can improve the performance of the classifier even if we don’t have a balanced dataset.

So, removing the Tomek links increases the gap between the two classes and thus facilitate the classification process.

In the following code, I will use ratio=majority to resample the majority class.

from imblearn.under_sampling import TomekLinks
tl = TomekLinks(return_indices=True, ratio='majority')
X_tl, y_tl, id_tl = tl.fit_sample(X, y)
print('Removed indexes:', id_tl)
Removed indexes: [     0      1      2 ... 284804 284805 284806]
12. Under-sampling : Cluster Centroids
In this technique, we perform under-sampling by generating centroids based on clustering methods. The dataset will be grouped by similarity, in order to preserve information.

In this example, I have passed the {0: 10} dict for the parameter ratio. It preserves 10 elements from the majority class (0), and all minority class (1) .

from imblearn.under_sampling import ClusterCentroids
cc = ClusterCentroids(ratio={0: 10})
X_cc, y_cc = cc.fit_sample(X, y)
print(X.shape[0] - X_cc.shape[0], 'New points undersampled under Cluster Centroids')
284305 New points undersampled under Cluster Centroids
13. Over-sampling : SMOTE
In the context of synthetic data generation, there is a powerful and widely used method known as synthetic minority oversampling technique or SMOTE. Under this technique, artificial data is created based on feature space. Artificial data is generated with bootstrapping and k-nearest neighbours algorithm. It works as follows:-

First of all, we take the difference between the feature vector (sample) under consideration and its nearest neighbour.
Then we multiply this difference by a random number between 0 and 1.
Then we add this number to the feature vector under consideration.
Thus we select a random point along the line segment between two specific features.
The concept of SMOTE can best be illustrated with the following figure:-

smote.png

So, SMOTE generates new observations by interpolation between existing observations in the dataset.

from imblearn.over_sampling import SMOTE
smote = SMOTE(ratio='minority')
X_sm, y_sm = smote.fit_sample(X, y)
print(X_sm.shape[0] - X.shape[0], 'New points created under SMOTE')
283823 New points created under SMOTE
14. Conclusion
In this jupyter notebook, I have discussed various approaches to deal with the problem of imbalanced classes. These are random oversampling, random undersampling, tree-based algorithms, resampling with imbalanced learn library, under-sampling : Tomek links, under-sampling : Cluster Centroids and over-sampling : SMOTE.

Some combination of these approaches will help us to create a better classifier. Simple sampling techniques may handle slight imbalance whereas more advanced methods like ensemble methods are required for extreme imbalances. The most effective technique will vary according to the dataset.

So, based on the above discussion, we can conclude that there is no one solution to deal with the imbalanced classes problem. We should try out multiple methods to select the best-suited sampling techniques for the dataset in hand. The most effective technique will vary according to the characteristics of the dataset.
