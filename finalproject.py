#Importing all the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


#Loading the dataset
data = pd.read_csv('Loan_Default.csv')

#Checking the head
data.head(5)

#Checking the tail
data.tail(5)

#Shape of the dataset
data.shape

#countplot to see the default status 0 is no and 1 is yes
sns.countplot(x='Status', data = data)

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Create a correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Choose a color palette and adjust transparency
sns.set_palette("coolwarm")
plt.figure(figsize=(10, 6))
sns.histplot(data['age'], bins=20, kde=True, alpha=0.7)  

# Add grid lines
plt.grid(True, alpha=0.3)

# Customize labels and title
plt.title('Age count as well as the frequency')
plt.xlabel('X-axis Label')
plt.ylabel('Frequency')

# Show the plot
plt.show()

#Crosstab for seeing the corelation between status and other columns
pd.crosstab(data.loan_limit,data.Status).plot(kind='bar')
plt.title('Column relation to the Status')
plt.xlabel('loan_limit')
plt.ylabel('Status')

pd.crosstab(data.approv_in_adv,data.Status).plot(kind='bar')
plt.title('Column relation to the Status')
plt.xlabel('approv_in_adv')
plt.ylabel('Status')

pd.crosstab(data.loan_type,data.Status).plot(kind='bar')
plt.title('Column relation to the Status')
plt.xlabel('loan_type')
plt.ylabel('Status')

# Create a Kernel Density Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=data, x='loan_amount', hue='Status', fill=True, common_norm=False, palette='Set1')
plt.title('Kernel Density Plot of Loan Amount by Status')
plt.xlabel('Loan Amount')
plt.ylabel('Density')
plt.legend(title='Status')
plt.grid(True)
plt.show()

# Create a count plot to visualize 'Status' by 'Region'
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Region', hue='Status', palette='Set2')
plt.title('Status Distribution by Region')
plt.xlabel('Region')
plt.ylabel('Count')
plt.legend(title='Status')
plt.grid(True)
plt.show()

# Create a box plot or violin plot to visualize 'Status' by 'age'
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Status', y='age', palette='Set1')
plt.title('Age Distribution by Status')
plt.xlabel('Status')
plt.ylabel('Age')
plt.grid(True)
plt.show()

#To get the unique value in each column
column_list = data.columns
for column_name in column_list:
    print(data[column_name].unique())

#Getting the data type for each column
data.dtypes

#getting a list for column with datatype object
obj=[]
for i in data.columns:
    if data[i].dtype=='object':
        obj.append(i)

for i in range(len(obj)):
    print(obj[i],data[obj[i]].unique())
    
for i in obj:
    print(i,'-',data[i].isnull().sum())
    
#Value count for 0 and 1 in status column
data['Status'].value_counts()

for i in range(len(obj)):
    print(obj[i],data[obj[i]].unique())

#Transforming the age from a series to a number
data['age'] = data['age'].replace({
    '35-44': 39.5,  # Mean age of 35-44 range
    '25-34': 29.5,  # Mean age of 25-34 range
    '55-64': 59.5,  # Mean age of 55-64 range
    '45-54': 49.5,  # Mean age of 45-54 range
    '65-74': 69.5,  # Mean age of 65-74 range
    '>74': 80,     # Assuming mean age of >74 range as 80
    '<25': 20      # Assuming mean age of <25 range as 20
})

#Dropping the irrelevant columns
data = data.drop(['ID','year','Gender','rate_of_interest','Interest_rate_spread','property_value','Upfront_charges','LTV','dtir1'],axis=1)

data = data.dropna()
data.isnull().sum()

data['Status'].value_counts()

obj=[]
for i in data.columns:
    if data[i].dtype=='object':
        obj.append(i)

l=[]
for i in obj:
    l.append(data.columns.get_loc(i))

#column split to get x and y variable for train test split
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

#Splitting the dataset 25/75 for train and test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

#Using one hot encoder for preprocessing
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),l)],remainder='passthrough')
x_train = ct.fit_transform(x_train)
x_test = ct.fit_transform(x_test)

#Creating a object from class
model1 = LogisticRegression()
model1.fit(x_train, y_train)

#Making predictions
y_pred = model1.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print('LogisticRegression')
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)

# Create a Decision Tree classifier
model2 = DecisionTreeClassifier(random_state=42)

# Train the model on the training data
model2.fit(x_train, y_train)

# Make predictions on the test data
y_pred = model2.predict(x_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print('DecisionTreeClassifier')
print("Accuracy:", accuracy * 100)
print("Precision:", precision * 100)
print("Recall:", recall * 100)
print("F1 Score:", f1 * 100)
print("ROC AUC Score:", roc_auc * 100)

# Create a Gaussian Naive Bayes classifier
model3 = GaussianNB()

# Train the model on the training data
model3.fit(x_train, y_train)

# Make predictions on the test data
y_pred = model3.predict(x_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print('Naive Bayes')
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)

# Model names
models = ['model1', 'model2', 'model3']

# Evaluation scores for each model
accuracy = [0.74, 0.74, 0.741]
precision = [0.00, 0.58, 0.378]
recall = [0.00, 0.62, 0.027]
f1 = [0.00, 0.60, 0.051]
roc_auc = [0.50, 0.73, 0.506]

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
plt.subplots_adjust(hspace=0.5)

# Bar plots for accuracy, precision, recall, F1 Score
axes[0].bar(models, accuracy, color='b', label='Accuracy')
axes[0].bar(models, precision, color='g', label='Precision')
axes[0].bar(models, recall, color='r', label='Recall')
axes[0].bar(models, f1, color='y', label='F1 Score')
axes[0].set_title('Comparison of Evaluation Metrics')
axes[0].legend()

# Bar plot for ROC AUC
axes[1].bar(models, roc_auc, color='purple', label='ROC AUC')
axes[1].set_title('Comparison of ROC AUC')
axes[1].legend()

# Show the plots
plt.show()

#Creating a link to app
joblib.dump(model2, 'decision_tree_model.pkl')


Link to the Data set: https://www.kaggle.com/datasets/yasserh/loan-default-dataset
H, M Yasser. “Loan Default Dataset.” Kaggle, 28 Jan. 2022, www.kaggle.com/datasets/yasserh/loan-default-dataset. 






