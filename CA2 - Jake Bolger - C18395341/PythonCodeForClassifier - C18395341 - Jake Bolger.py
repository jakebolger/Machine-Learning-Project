##TITLE: - CA2 - Build a Classifier
##AUTHOR: Jake Bolger
##STUDENT NUMBER: C18395341
##COURSE: TU857

##DESCRIPTION: In this assignment I will develop a classifier that uses data to predict the outcome of a bank marketing campaign. The classifier model has to be one of those studied in this course.
##DATA AVAILABLE: 'datadescription.txt', 'trainingset.txt', 'queries.txt'
#CLASSIFIER: The classifier used was the KNN classifier.
#Written in Jupyter Notebook
#

#importing pandas
#
import pandas as pd

#serializing and de-serializing python object structures.
#
import pickle

#cross_validation doesnet work anymore has been depricated
#
from sklearn.model_selection import train_test_split

#from sklearn.cross_validation import train_test_split
#
from sklearn.metrics import confusion_matrix

#python library for list related operations
#
import numpy as np

#importing model selection
#
from sklearn import model_selection

#importing KNeighbors
#
from sklearn.neighbors import KNeighborsClassifier

#importing stratifiedKfold
#
from sklearn.model_selection import StratifiedKFold

#matplotlib to plot the data
#
import matplotlib.pyplot as plt
%matplotlib inline

#importing preprocessing
#
from sklearn import preprocessing

#importing roc_curve and auc
#
from sklearn.metrics import roc_curve, auc

#importing accuracy score
#
from sklearn.metrics import accuracy_score


#Creating an array for the my_headers_array, this will allow the names to be set as headers.
#
my_headers_array = []

#Opening the givin text file 'datadescription.txt' and reading it in.
#When running on different machine make sure to change directory location.
#
with open('C:/Users/EKAJ/Documents/Machine Learning/CA2 - Build a Classifier - Jake Bolger/data/datadescription.txt', 'r') as dd:
    #Extracting all the column names from the file
    #
    for line in dd:
        if line[0].isdigit():
            items = line.split(' ')
            my_headers_array.append(items[2].strip().replace(':', ''))
        #End if
        #
    #End For
    #


#reading in the file 'trainingset.txt' and setting it to 'myTrainingSet'.
#
myTrainingSet = pd.read_csv('C:/Users/EKAJ/Documents/Machine Learning/CA2 - Build a Classifier - Jake Bolger/data/trainingset.txt', header=None)

#Sets the headers from the array to data
#
myTrainingSet.columns = my_headers_array    


#setting the headers of the categorical columns
#
column_cat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'output']

#dummies function to convert data
#
dummies_myTrainingSet = pd.get_dummies(myTrainingSet, columns = column_cat )

#export to csv file 'headersTrainingSet.csv'
#
myTrainingSet.to_csv('C:/Users/EKAJ/Documents/Machine Learning/CA2 - Build a Classifier - Jake Bolger/data/headersTrainingSet.csv', index=False)


#read in the csv file with headers
#
trainingsetData = pd.read_csv('C:/Users/EKAJ/Documents/Machine Learning/CA2 - Build a Classifier - Jake Bolger/data/headersTrainingSet.csv') 
#


#get the output features 
#
selectedHeader = trainingsetData['output']

#create cvs file called 'features_file.csv' with all the outputs.
#
selectedHeader.to_csv('C:/Users/EKAJ/Documents/Machine Learning/CA2 - Build a Classifier - Jake Bolger/data/features_file.csv', index=False)


#setting the headers of the numerical columns
#
column_numeric = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

#setting to numerical_ts
#
numerical_ts = trainingsetData[column_numeric]


#dropping the columns 'id', and 'output'.
#
categorical_ts = trainingsetData.drop(column_numeric + ['id','output'], axis=1)


# replacing the empty values
#
categorical_ts.replace('?', 'NA')

#adding NA
#
categorical_ts.fillna('NA', inplace=True)


#use get_dummies to switch to numerical
#
categoricalNew = pd.get_dummies(categorical_ts)


#combining the numerical and categorical features.
#
combined_ts = np.hstack((numerical_ts.values, categoricalNew))



#storing data in 'myData' file.
#
np.save('C:/Users/EKAJ/Documents/Machine Learning/CA2 - Build a Classifier - Jake Bolger/data/myData', combined_ts)



#trainin gin the model with 'myDataFile', and 'featureFileData'.
#'myData.npy'
#
myDataFile = np.load('C:/Users/EKAJ/Documents/Machine Learning/CA2 - Build a Classifier - Jake Bolger/data/myData.npy')

#'features_file.csv'
#
featureFileData = pd.read_csv('C:/Users/EKAJ/Documents/Machine Learning/CA2 - Build a Classifier - Jake Bolger/data/features_file.csv')


#switching to binary layout
#
processor = preprocessing.LabelEncoder()

#fit
#
processor.fit(featureFileData)

#transform
#
featureFileData = processor.transform(featureFileData)

#creating vector
#
my_vector = np.linspace(0, 1, 101)


#--KNN (K-Nearest-Neighbors)--
#Using KNN classifier section
#

#importing GaussianNB
#
from sklearn.naive_bayes import GaussianNB

#setting stratified to skf
#
my_strat = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

#using get_n_splits
#
my_strat.get_n_splits(myDataFile, featureFileData)

#array to hold data
#
masterArray = []

#for loop for index variables for training data
#
for i, j in my_strat.split(myDataFile, featureFileData):
    
    #create KNN
    #
    KNN_Classifier = KNeighborsClassifier(n_neighbors=10)
    
    #integrate the model
    #
    KNN_Classifier.fit(myDataFile[i], featureFileData[i])
    
    #creating outcomes
    #
    outcomes = KNN_Classifier.predict(myDataFile[j])
    
    #printing the accuracy of model
    #
    #print('Accuracy= ' + str(accuracy_score(featureFileData[j], outcomes, normalize=True)))
    
    #Use probability counters
    #
    unique, counts = np.unique(outcomes, return_counts = True)
    
    #use dictionary, zip with counters
    #
    variable_master = dict(zip(unique, counts))
    
    #printing "Count"
    #
    #print("Number" + str(variable_master))
    
    #Making the ROC curve graph Plot
    #
    fpr, tpr, _ = roc_curve(featureFileData[j], outcomes[:])
    
    #printing the Area.
    #
    #print("Curve" + str(auc(fpr, tpr)))
    
    
    #using interp
    #
    tpr = np.interp(my_vector, fpr, tpr)
    
    #setting to 0.0
    #
    tpr[0] = 0.0
    
    #appending the array
    #
    masterArray.append(tpr)
    #End
    #

#processing array using numpy
#
masterArray = np.array(masterArray)

#Getting Mean
#
mean_tprs = masterArray.mean(axis=0)

#Getting std
#
std = masterArray.std(axis=0)

#Getting minimum
#
tprs_upper = np.minimum(mean_tprs + std, 1)

#taking away std
#
tprs_lower = mean_tprs - std

#All the plot code below was used to make accuracy data and plot data in report not needed for program*
#--PLOT--
#This part is making the 2d plot
#
#plt.plot(my_vector, mean_tprs, 'b')

#using fill_between and setting colour
#
#plt.fill_between(my_vector, tprs_lower, tprs_upper, color='black', alpha=0.3)




#making an instance for the KNN
#
KNN_Classifier = KNeighborsClassifier(n_neighbors=10)

#using fit to add the model
#
KNN_Classifier.fit(myDataFile, featureFileData)

#using dump and saving the model that was created
#
pickle.dump(KNN_Classifier, open('myModel.sav', 'wb'))


#readin in the givin text file 'queries.txt'
#
myTrainingSet = pd.read_csv('C:/Users/EKAJ/Documents/Machine Learning/CA2 - Build a Classifier - Jake Bolger/data/queries.txt', header=None)

#Adding the column names to trainingset data.
#
myTrainingSet.columns = my_headers_array

#creating csv file for queries file with headers
#
myTrainingSet.to_csv('C:/Users/EKAJ/Documents/Machine Learning/CA2 - Build a Classifier - Jake Bolger/data/queriesHeaders.csv', index=False)



#reading in 'queriesHeaders.csv' file
#
tsQueries = pd.read_csv('C:/Users/EKAJ/Documents/Machine Learning/CA2 - Build a Classifier - Jake Bolger/data/queriesHeaders.csv') 

#setting the names of numeric features
#
column_numeric = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

#setting to numerical_ts
#
numerical_ts = tsQueries[column_numeric]


#separating the 'id'.
#
myId = tsQueries['id']


#dropping the 'id', and 'output'.
#
categorical_ts = tsQueries.drop(column_numeric + ['id','output'], axis=1)


#replacing empties
#
categorical_ts.replace('?', 'NA')

#filling with NA
#
categorical_ts.fillna('NA', inplace=True)


#use get_dummies to switch to numerical
#
categoricalNew = pd.get_dummies(categorical_ts)


#combining the numerical and categorical features.
#
combined_ts = np.hstack((numerical_ts.values, categoricalNew))


#using 'load' to use the model and get the outcomes
#
useModel = pickle.load(open('myModel.sav', 'rb'))

#create testing outcomes
#
outcomes = useModel.predict(combined_ts)


#export outcomes to text file 'outcome.txt'
#
myId.to_csv('C:/Users/EKAJ/Documents/Machine Learning/CA2 - Build a Classifier - Jake Bolger/data/outcomes.txt', index=False)


#Creating i variable and set to 0
#
i = 0

#Open 'solutionFile.txt' file.
#
with open('C:/Users/EKAJ/Documents/Machine Learning/CA2 - Build a Classifier - Jake Bolger/data/solutionFile.txt', 'a') as pred_file:
    
    #Open 'outcomes.txt' file.
    #
    with open('C:/Users/EKAJ/Documents/Machine Learning/CA2 - Build a Classifier - Jake Bolger/data/outcomes.txt', 'r+') as res_file:
        
        #for loop and if statement to output predictions to solution file 'solutionFile.txt'.
        #
        for line in res_file:
            
            #for typeA
            #
            if outcomes[i] == 0:
                pred_file.write(line.strip() + ',"TypeA"\n')
            
            #else typeB
            #
            else:
                pred_file.write(line.strip() + ',"TypeB"\n')
            i += 1
            #End if, else
            #
        #End for
        #