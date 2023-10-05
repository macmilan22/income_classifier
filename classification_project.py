# =============================================================================
# CLASSIFYING PERSONAL INCOME 
# =============================================================================
################################# Required packages ############################






# To work with dataframes
import pandas as pd 

# To perform numerical operations
import numpy as np

# To visualize data
import seaborn as sns
import matplotlib.pyplot as plt

# To partition the data
from sklearn.model_selection import train_test_split

# Importing library for logistic regression
from sklearn.linear_model import LogisticRegression

# Importing performance metrics - accuracy score & confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix




###############################################################################
# =============================================================================
# Importing data
# =============================================================================




data_income = pd.read_csv('C:\\Users\\HP\\OneDrive\\Desktop\\New folder (4)\\income1.csv')                                         #,na_values=[" ?"]) 
  
# Creating a copy of original data                                                                              # Additional strings (" ?") to recognize as NA
data = data_income.copy()

"""
#Exploratory data analysis:

#1.Getting to know the data
#2.Data preprocessing (Missing values)
#3.Cross tables and data visualization
"""



# =============================================================================
# Getting to know the data
# =============================================================================



#**** To check variables' data type
print(data.info())

#**** Check for missing values             
data.isnull()          
       
print('Data columns with null values:\n', data.isnull().sum())
#**** No missing values !

#**** Summary of numerical variables
summary_num = data.describe()
print(summary_num)            

#**** Summary of categorical variables
summary_cate = data.describe(include = "O")
print(summary_cate)

#**** Frequency of each categories
data['JobType'].value_counts()
data['occupation'].value_counts()

#**** Checking for unique classes
print(np.unique(data['JobType'])) 
print(np.unique(data['occupation']))
#**** There exists ' ?' instesd of nan

"""
Go back and read the data by including "na_values[' ?']" to consider ' ?' as nan !!!
"""
data = pd.read_csv('C:\\Users\\HP\\OneDrive\\Desktop\\New folder (4)\\income1.csv',na_values=[" ?"]) 




# =============================================================================
# Data pre-processing
# =============================================================================



data.isnull().sum()

missing = data[data.isnull().any(axis=1)]
# axis=1 => to consider at least one column value is missing in a row

""" Points to note:
1. Missing values in Jobtype    = 1809
2. Missing values in Occupation = 1816 
3. There are 1809 rows where two specific 
   columns i.e. occupation & JobType have missing values
4. (1816-1809) = 7 => You still have occupation unfilled for 
   these 7 rows. Because, jobtype is Never worked
"""

data2 = data.dropna(axis=0)
data3 = data2.copy()
data4 = data3.copy()
# Realtionship between independent variables
correlation = data2.corr()




# =============================================================================
# Cross tables & Data Visualization
# =============================================================================


# Extracting the column names
data2.columns   



# =============================================================================
# Gender proportion table:
# =============================================================================



gender = pd.crosstab(index = data2["gender"],
                     columns  = 'count', 
                     normalize = True)
print(gender)


# =============================================================================
#  Gender vs Salary Status:
# =============================================================================


gender_salstat = pd.crosstab(index = data2["gender"],
                             columns = data2['SalStat'], 
                             margins = True, 
                             normalize =  'index') # Include row and column totals
print(gender_salstat)




# =============================================================================
# Frequency distribution of 'Salary status' 
# =============================================================================




# Create a countplot for the 'SalStat' column
SalStat = sns.countplot(data=data2, x='SalStat')

# Add labels and a title
plt.xlabel('SalStat')
plt.ylabel('Count')
plt.title('Distribution of SalStat')


"""  75 % of people's salary status is <=50,000 
     & 25% of people's salary status is > 50,000
"""

##############  Histogram of Age  #############################
sns.distplot(data2['age'], bins=10, kde=False)
# People with age 20-45 age are high in frequency

############# Box Plot - Age vs Salary status #################
# Assuming 'SalStat' and 'age' are columns in your 'data2' DataFrame
sns.boxplot(x='SalStat', y='age', data=data2)

# Optionally, display the plot
plt.show()


## x='SalStat' specifies that the 'SalStat' column should be used for the x-axis (typically the categorical variable).
## y='age' specifies that the 'age' column should be used for the y-axis (typically the numerical variable).
## data=data2 specifies the DataFrame containing your data.



 
## people with 35-50 age are more likely to earn > 50000 USD p.a
## people with 25-35 age are more likely to earn <= 50000 USD p.a

#*** Jobtype
JobType     = sns.countplot(y=data2['JobType'],hue = 'SalStat', data=data2)
job_salstat =pd.crosstab(index = data2["JobType"],columns = data2['SalStat'], margins = True, normalize =  'index')  
round(job_salstat*100,1)


#*** Education
Education   = sns.countplot(y=data2['EdType'],hue = 'SalStat', data=data2)
EdType_salstat = pd.crosstab(index = data2["EdType"], columns = data2['SalStat'],margins = True,normalize ='index')  
round(EdType_salstat*100,1)

#*** Occupation
Occupation  = sns.countplot(y=data2['occupation'],hue = 'SalStat', data=data2)
occ_salstat = pd.crosstab(index = data2["occupation"], columns =data2['SalStat'],margins = True,normalize = 'index')  
round(occ_salstat*100,1)

#*** Capital gain
# it can be an important variable as if capital gain is high salstat can be high
# plot shows 92% of the capital gain is 0 which corresponds to 27611 observations and on;y 8% of people have gained profit out of their investments
sns.distplot(data2['capitalgain'], bins = 10, kde = False)

# 95% of the capital loss is 0 i.e. 28721 observations
# lack of investment can be the reason of no capital loss

sns.distplot(data2['capitalloss'], bins = 10, kde = False)

#*** Hours spent

# Create the boxplot

sns.boxplot(x='SalStat', y='hoursperweek', data=data2)

# Optionally, customize the plot appearance
plt.xlabel('Salary Status (0: <50000, 1: >50000)')
plt.ylabel('Hours per Week Worked')
plt.title('Boxplot of Salary Status by Hours per Week Worked')
# Display the plot
plt.show()







# =============================================================================
# LOGISTIC REGRESSION
# =============================================================================



# Reindexing the salary status names to 0,1
del data2

data2= data3.copy()
data3['SalStat']=data3['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
 
new_data=pd.get_dummies(data2, drop_first=True)

# Storing the column names 
columns_list=list(new_data.columns)
print(columns_list)

# Separating the input names from data
features=list(set(columns_list)-set(['SalStat']))
print(features)

# Storing the output values in y
y=new_data['SalStat'].values
print(y)

# Storing the values from input features
x = new_data[features].values
print(x)

# Splitting the data into train and test
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)

# Make an instance of the Model
logistic = LogisticRegression()

# Fitting the values for x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

# Prediction from test data
prediction = logistic.predict(test_x)
print(prediction)

# Confusion matrix
confusion_matrix = confusion_matrix(test_y, prediction)
print(confusion_matrix)

# Calculating the accuracy
from sklearn.metrics import accuracy_score
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)

# Printing the misclassified values from prediction

print('Misclassified samples: %d' % (test_y != prediction).sum())



# =============================================================================
# LOGISTIC REGRESSION - REMOVING INSIGNIFICANT VARIABLES
# =============================================================================



# Reindexing the salary status names to 0,1
data4['SalStat']=data4['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data4['SalStat'])

cols = ['gender','nativecountry','race','JobType']
new_data = data4.drop(cols,axis = 1)

new_data=pd.get_dummies(new_data, drop_first=True)

# Storing the column names 
columns_list2=list(new_data.columns)
print(columns_list2)

# Separating the input names from data
features2=list(set(columns_list2)-set(['SalStat']))
print(features2)

# Storing the output values in y
y2=new_data['SalStat'].values
print(y2)

# Storing the values from input features
x2 = new_data[features2].values
print(x2)

# Splitting the data into train and test
train_x2,test_x2,train_y2,test_y2 = train_test_split(x2,y2,test_size=0.3, random_state=0)

# Make an instance of the Model
logistic2 = LogisticRegression()

# Fitting the values for x and y
logistic2.fit(train_x2,train_y2)

# Prediction from test data
prediction2 = logistic2.predict(test_x2)

# calculating the accuracy 
from sklearn.metrics import accuracy_score
accuracy_score=accuracy_score(test_y2, prediction2)
print(accuracy_score)

# Printing the misclassified values from prediction
print('Misclassified samples: %d' % (test_y2 != prediction2).sum())





# =============================================================================
# KNN
# =============================================================================


# importing the library of KNN
from sklearn.neighbors import KNeighborsClassifier


# Storing the K nearest neighbors classifier
KNN_classifier = KNeighborsClassifier(n_neighbors = 5)  

# Fitting the values for X and Y
KNN_classifier.fit(train_x, train_y) 

# Predicting the test values with model
prediction = KNN_classifier.predict(test_x)

# Performance metric check
from sklearn.metrics import confusion_matrix

# Create a confusion matrix and assign it to a variable
confusionMmatrix = confusion_matrix(test_y, prediction)

# Later in the code, you mistakenly try to call confusionMmatrix as if it were a function
print(confusionMmatrix(test_y, prediction))

# Calculating the accuracy
accuracy_score3=accuracy_score(test_y, prediction)
print(accuracy_score)

print('Misclassified samples: %d' % (test_y != prediction).sum())

"""
Effect of K value on classifier
"""
Misclassified_sample = []
# Calculating error for K values between 1 and 20
for i in range(1, 20):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x, train_y)
    pred_i = knn.predict(test_x)
    Misclassified_sample.append((test_y != pred_i).sum())

print(Misclassified_sample)




# =============================================================================
# END OF SCRIPT
# =============================================================================