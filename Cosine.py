import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Adjusted Dry Bean Dataset.csv')
X = df[['Eccentricity', 'Solidity', 'Compactness', 'roundness', 'Area']]
y = df[['Class']]
print(X.isna().sum())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
# Setting the number of neighbors
classifier = KNeighborsClassifier(n_neighbors=10, metric = "euclidean")
# Loading the training set
classifier.fit(X_train, np.ravel(y_train,order='C'))

# Predicting the test labels
y_pred = classifier.predict(X_test)
print(y_pred)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# creating confusion matrix and printing the classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
accuracy =  accuracy_score(y_test,y_pred)*100
print(accuracy)

k_range = range(1, 40)

# Creating a Python dictionary by [] and then appending the accuracy scores

scores = []
#  looping through the k range 1 to 40

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,np.ravel(y_train,order='C'))
    y_pred = knn.predict(X_test)
    # appending the accuracy scores in the dictionary named scores.
    scores.append(metrics.accuracy_score(y_test, y_pred))

print(scores)
# Printing the K number of neighbors and Testing Accuracy.
import matplotlib.pyplot as plt

# This command allow plots to appear within the notebook
plt.plot(k_range, scores)
plt.xlabel('Value of K')
plt.ylabel('Testing Accuracy')
plt.savefig(fname = "Testing_Accuracy.png", transparent = True)
plt.show()

# Optimizing the k-nn by using Cross validation
from sklearn.model_selection import cross_val_score
import numpy as np
#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=15)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)

X = scaler.transform(X)

#train model with cv of 10
cv_scores = cross_val_score(knn_cv, X, np.ravel(y,order='C'), cv=10)
#print each cv score (accuracy) and average them
print(cv_scores)
print(np.mean(cv_scores))

# Using cross validation with all possible k values.
from sklearn.model_selection import cross_val_score
import numpy as np
#create a new KNN model

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)

# Train with 10 fold cross validation by an outer k value ranges and nested cross validation scores.
X = scaler.transform(X)
scores = []
k_range = range(1, 40)
for k in k_range:
#train model with cv of 10
    knn_cv = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(knn_cv, X, np.ravel(y,order='C'), cv=10)
#print each cv score (accuracy) and average them
    print(k)
    print(cv_scores)
    print(np.mean(cv_scores))
    
# Prediction     
knn = KNeighborsClassifier(n_neighbors=14)
knn.fit(X_train,np.ravel(y_train,order='C'))
y_pred = knn.predict(X_test)
accuracy_scores = metrics.accuracy_score(y_test, y_pred) 
print(accuracy_scores) 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Adjusted Dry Bean Dataset.csv')

# Exclude the 'Class' column which contains string values and the first column if it's an index
X = df.iloc[:, 3:13]  # Adjust the range if needed
y = df['Class Number']  # Make sure this column is numeric

print(X)
print(y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)


# Random Forest to get feature importance for classification
rf = RandomForestClassifier(n_estimators=100, random_state=12)
rf.fit(X_train, y_train)

# Get the feature importances and sort them
sorted_idx = rf.feature_importances_.argsort()
print("Feature importances:", rf.feature_importances_)


# Print sorted indices (i.e., the indices of the features in ascending order of importance)
print("Sorted indices of features:", sorted_idx)

# Plotting feature importances
plt.barh(X_train.columns[sorted_idx], rf.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
plt.savefig(fname = "Forest_Feauture_Importance.png", transparent = True)
plt.show()


# Using a subset of predictor feature variables for the classification:
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
#create a new KNN model

from sklearn.preprocessing import StandardScaler
from sklearn import metrics

data_Optimize = pd.read_csv('Adjusted Dry Bean DataSet.csv')
# Remove spaces in the column names
data_Optimize.columns = data_Optimize.columns.to_series().apply(lambda x: x.strip())
X = data_Optimize[['Area','roundness','Compactness','Solidity','Eccentricity',]] 
y = df[['Class']]


scaler = StandardScaler()
scaler.fit(X)

X = scaler.transform(X)
scores = []
k_range = range(1, 40)
for k in k_range:
#train model with cv of 10
    knn_cv = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(knn_cv, X, np.ravel(y,order='C'), cv=10)
#print each cv score (accuracy) and average them
    print(k)
    print(cv_scores)
    print(np.mean(cv_scores))
    
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train,np.ravel(y_train,order='C'))
y_pred = knn.predict(X_test)
accuracy_scores = metrics.accuracy_score(y_test, y_pred) 
print(accuracy_scores)  


from sklearn.neighbors import KNeighborsClassifier
test_scores = []
train_scores = []

for i in range(1,15):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))
    
## Training Evaluation
max_train_score = max(train_scores)

# # Store the max train test score index by enumerating through all the scores.

train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]

# Store the max score in the first curly parenthesis and the indices in the second.
# The lambda function takes the index starting at zero therefore one is added to get the k value.

print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))

## Testing Evaluation
max_test_score = max(test_scores)

test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))

## Train Test Evaluation by comparative graph.
plt.figure(figsize=(12,5))
p = sns.lineplot(x=range(1,15), y=train_scores, marker='*', label='Train Score')
p = sns.lineplot(x=range(1,15), y=test_scores, marker='o', label='Test Score')
plt.savefig(fname = "Train_Test_LinePlot.png", transparent = True)
plt.show()


## Error Rate Graph
# Create an empty dictionary to collect errors across the different k-values
error = []
# Iterate through k=1 to 40  and run the classifier.Predict and append the error for each iteration.
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

    # Create a plot of Mean error versus kvalue.
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.savefig(fname = "Error_Rate.png", transparent = True)
plt.show()