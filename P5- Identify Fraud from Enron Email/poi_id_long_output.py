#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")


from sklearn.model_selection import train_test_split 
from sklearn.metrics import recall_score 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score 
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


from time import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing other helper functions created through Udacity
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# 16 total original features not including POI 
# 2 additional features added 
features_list = ['poi',\
'salary', \
'from_this_person_to_poi', 'from_poi_to_this_person', \
'long_term_incentive', 'total_payments', \
'bonus', 'deferral_payments', 'loan_advances', \
'restricted_stock_deferred', 'deferred_income', 'total_stock_value', \
'expenses', 'exercised_stock_options', 'other', \
'restricted_stock', 'director_fees', \
#created features:
"fraction_from_poi", "fraction_to_poi"] 

# You will need to use more than just salary 

print "Number of features in original features_list = ", len(features_list)
print 
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#Exploration: 

#print "Park outlier "
#print data_dict['THE TRAVEL AGENCY IN THE PARK']
#print 

print "Number of people in dataset = ", len(data_dict)
print 

# cycle through the dataset to find pois and non-pois
poinum = 0 
poilist = []
nonpois = []
for person in data_dict.keys():
    #if the person is a poi, increment poi by 1 and add their name to 
    #the list 
    if data_dict[person]["poi"]==1:
        poinum += 1 
        poilist.append(person)
    #if a person isn't a poi, add them to the non-poi list 
    else: 
        nonpois.append(person)

print "Number of POIs = " , poinum 
print 
#number of non-pois is the length of the list of non-pois
nonnum = len(nonpois)
print "Number of non POIs =", nonnum

print 
#print "keys = ", data_dict.keys()
#print 
#print 
#print "values = ", data_dict.values()[0]

#Original Variables of the data_dict 
origvars = data_dict.values()[0].keys()
#print "Original variables = ", origvars


#MISSING VARIABLE INVESTIGATION
#Initialize dictionaries to keep track of missing variables
#General Data
missing = {}
#POIs
nanpoi = {}
#Non POIs
nannon = {}
# add the variables as keys to the dictionaries 
# with an initial count of 0 
for var in origvars:
    missing[var] = 0 
    nanpoi[var] = 0 
    nannon[var] = 0 

# For each person in the data_dict, count the variables that are missing 
# and increment each respectible count by 1 
for observation in data_dict.values(): 
    for variable in observation: 
        #print "var = ", variable 
        value = observation[variable]
        #For the General Data/Total 
        if value == "NaN":
            missing[variable]+= 1 
    #If someone is a POI
    if observation['poi'] == 1: 
        for variable in observation: 
            value = observation[variable]
            if value == "NaN": 
                nanpoi[variable] +=1
    else: # If someone isn't a POI
        for variable in observation: 
            value = observation[variable]
            if value == "NaN": 
                nannon[variable] +=1

print "Number of Missing Variables for each Group" 
print "All Data: "
print missing 
print 
#print "POIs"
#print nanpoi
print 
#print "Non-POIs"
#print nannon
print 

#Initialize a dictionary of empty dictionaries 
#This will hold the missing data overall as well 
#as by category (poi/non)
aggmiss = {'Overall': {}, 'POIs': {}, 'Non-POIs': {}, 'Ratio': {}}

for category in origvars: 
    #for each variable in the list of original variables 
    #get the number of missing items for the general data 
    total = missing[category]
    #number of missing items for POIs
    poimis = nanpoi[category]
    #number of missing items for Non POIs
    nonmis = nannon[category]
    #Add the proportion of data missing from each variable 
    aggmiss['Overall'][category] = float(total)/float(len(data_dict))
    #Add the proportion of missing data for POIs 
    aggmiss['POIs'][category] = float(poimis)/float(poinum)
    #Add the proportion of missing data for NonPois 
    aggmiss['Non-POIs'][category] = float(nonmis)/float(nonnum)
    #Add the ratio of missing data for POIs to non POIs
    #accounting for 0 values and dividing by 0
    if float(nonmis) > 0: 
        aggmiss['Ratio'][category] = float(poimis)/float(nonmis)
    else: 
        aggmiss['Ratio'][category] = 'N/A'

#print "Proportion of data missing"
#print "Aggregate" 
#print aggmiss['Overall']
#print 
#print "POIs"
#print aggmiss['POIs']
#print "Non-POIs"
#print aggmiss['Non-POIs']
#print "Ratio"
#print aggmiss['Ratio']


### Task 2: Remove outliers
# removing outliers from the data
# Total line was the total line from the financial spreadsheet 
# The Travel Agency in the Park was actually a Travel Agency 
# not of interest  
print "Removing Outliers"
print 
outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
for outlier in outliers: 
    data_dict.pop(outlier, 0 )

### Task 3: Create new feature(s)
# From Feature Selection lession 

print "Creating New Features"
print 

#Helper function from Udacity lesson
def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    ### 	returns either
    ###     the fraction of all messages to this person that come from POIs
    ###     or
    ###     the fraction of all messages from this person that are sent to POIs
    ### the same code can be used to compute either quantity

    ### takes into account"NaN" when there is no known email address (and so
    ### no filled email features), and integer division!
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    fraction = 0.
    if poi_messages != "NaN": 
        if all_messages!= "NaN": 
            fraction = float(float(poi_messages)/float(all_messages))

    return fraction


# Now, finding the fractions for each person in our dataset 
for name in data_dict.keys():

    data_point = data_dict[name]
    # get the necessary variables from each persons info 
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    # use the helper function to calculate the fraction 
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    # add the fraction to the persons information in our data_dict 
    data_point["fraction_from_poi"] = fraction_from_poi
    # repeat this process for the from person to poi 
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )  
    data_point["fraction_to_poi"] = fraction_to_poi

### Store to my_dataset for easy export below.
my_dataset = data_dict
# testing the available features
#print "Values = ", my_dataset.values()[0] 
print "New number of available features = ", len(my_dataset.values()[0])
print 
print "Mark Metts example = ", my_dataset['METTS MARK']
print 
#print "Number of features = ", len(my_dataset['METTS MARK'])
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#print "After extraction"
#print "Length of first item in features)", len(features[0])


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Note: Various Classifiers below 

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Create a helper function to evaluate algorithms 
def evaluate(features_test, labels_test):
    pred = clf.predict(features_test)
    # Accuracy 
    accuracy = accuracy_score(labels_test, pred)
    print "Accuracy = ", accuracy 
    # Precision 
    precision = precision_score(labels_test, pred)
    print "Precision = ", precision
    # Recall 
    recall = recall_score(labels_test, pred)
    print "Recall = ", recall 
    # F1 
    f1 = f1_score(labels_test, pred)
    print "F1 = ", f1 
    print 
    return 

sss = StratifiedShuffleSplit(n_splits= 10, random_state=42)

# use train test split to split the data between training and testing sets 
print "Splitting Data for Validation"
print 
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# print the length of the features 
print "Number of observations = ", len(features)
print "Number in training set = ", len(features_train)
print "Number in testing set = ", len(features_test)
print 

print "Experimental Gaussian NB Model starts here"
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
#Get Accuracy, Precision, and Recall 
print "Performance using all variables"
evaluate(features_test, labels_test)

print "Class Probabilities:"
print clf.class_prior_
print 

#Scale Features 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

newfeatures = np.array(features_train)
print "Scaling Features..."
print
# scale the features so one doesn't outweigh the others 
# this is important for algorithms computing distance 
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

print "Successfully Scaled Features" 
print "Example from new features: " 
print newfeatures[0]

#Explore Features post scaling using Decision Tree 

print "Exploratory DT starts here "
#from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split = 15, \
    class_weight = 'balanced')
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print "Feature Importances:"
print clf.feature_importances_

print "Decision Path"
print clf.decision_path

#Get Accuracy, Precision, and Recall 
evaluate(features_test, labels_test)

# Perform PCA 
print "PCA Starts Here:"
#from sklearn.decomposition import PCA
pca = PCA(n_components=1).fit(features_train)

print "Eigenvalues and explained_variance_ratio_"
#this is how you know how much variation of the data is in each component 
print pca.explained_variance_ratio_

#the direction of the new components in the new feature space 
#xprime in new feature space 

first_pc = pca.components_[0]
#second_pc = pca.components_[1]

print "first_pc", first_pc
print 
#print "second_pc", second_pc
print 
#transformed_data = pca.transform(data)
# get the transformed data times the direction of the new components to plot 

#print "Projecting the input data on the eigenfaces orthonormal basis"
X_train_pca = pca.transform(features_train)
X_test_pca = pca.transform(features_test)

print "Data transformed"
#print X_test_pca

#print "Are they the same: ", X_test_pca==features_test
#print 
print "Pre PCA feature example:"
print features_test[0]
print 
print "Post PCA feature example"
print X_test_pca[0]
print 


# Fit the NB model 
print "GaussianNB STARTS HERE "
clf = GaussianNB()
clf.fit(features_train, labels_train)
#Get Accuracy, Precision, and Recall 
evaluate(features_test, labels_test)


#Now trying using PCA options 
print "PCA Set:"
clf.fit(X_train_pca, labels_train)

#Get Accuracy, Precision, and Recall 
evaluate(X_test_pca, labels_test)

# Ridge Classification 
print "Ridge Classification Starts Here:"
from sklearn import linear_model
clf = linear_model.RidgeClassifier(class_weight = 'balanced')
clf.fit(features_train, labels_train)
#print "Coefficients: " 
#print clf.coef_
#print 
#print "Ridge Classification Predictions"
#pred = clf.predict(features_test)
#print pred
#print 
#print "True values/labels"
#print labels_test
#print 
#print "Which predicted values match?"
#print pred==labels_test
print 
#Get Accuracy, Precision, and Recall 
evaluate(features_test, labels_test)

#Now trying using PCA options 
print "PCA Set:"
clf.fit(X_train_pca, labels_train)

#Get Accuracy, Precision, and Recall 
evaluate(X_test_pca, labels_test)


print "DT Classifier starts here "
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split = 20, \
    class_weight = 'balanced')
clf = clf.fit(features_train, labels_train)

print "Feature Importances:"
print clf.feature_importances_

#Get Accuracy, Precision, and Recall 
evaluate(features_test, labels_test)

#Now trying using PCA options 
print "PCA Set:"
clf.fit(X_train_pca, labels_train)

#Get Accuracy, Precision, and Recall 
evaluate(X_test_pca, labels_test)
print 


print "RadiusNeighborsClassifier starts here"
from sklearn.neighbors import RadiusNeighborsClassifier
clf = RadiusNeighborsClassifier(weights = 'distance', \
    radius = .4, outlier_label = 1)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print "Radius predictions"
print pred 
#Get Accuracy, Precision, and Recall 
evaluate(features_test, labels_test)

#Now trying using PCA options 
print "PCA Set:"
clf.fit(X_train_pca, labels_train)

#Get Accuracy, Precision, and Recall 
evaluate(X_test_pca, labels_test)

# Now trying again using Pipelines 
print "SVC PIPELINE STARTS HERE "

from sklearn.pipeline import Pipeline
estimators = [('reduce_dim', PCA()), ('clf', \
        SVC(class_weight= 'balanced'))]
pipe = Pipeline(estimators)

param_grid = dict(reduce_dim__n_components=[1, 2, 3],\
        clf__C=[ 1, 3, 5, 7, 10], 
        clf__gamma =[0.01, 0.1, .3, .5, 1, 5, 10])

grid_search = GridSearchCV(pipe, param_grid=param_grid, \
    scoring = 'f1', cv = sss, iid = False)

grid_search.fit(features_train, labels_train)

print "grid Score = ", grid_search.score(features_test, labels_test)
print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
#print "Best parameters"
print 
print "GridSearch  "
print(grid_search.best_params_)
print 
pred2 = grid_search.predict(features_test)
#print "PREDICTIONS HERE"
#print pred2

#print "Classification report for SVC with PCA and GridSearch and using Pipeline"
#print classification_report(labels_test, pred2)

# Accuracy 
accuracy = accuracy_score(labels_test, pred2)
print "Accuracy = ", accuracy 
# Precision 
precision = precision_score(labels_test, pred2)
print "Precision = ", precision
# Recall 
recall = recall_score(labels_test, pred2)
print "Recall = ", recall 
print 
# F1 
f1 = f1_score(labels_test, pred2)
print "F1 = ", f1 
print 

print "FINAL Classifier starts here: SVC "
#create the classifier
clf = SVC(C = 1, kernel = 'rbf', gamma = .3, class_weight = 'balanced')
#Now trying using PCA options 
print "PCA Set:"

#time the training
t0 = time()
clf.fit(X_train_pca, labels_train)
print "training time:", round(time()-t0, 3), "s"

#time the prediction
t1 = time()
pred = clf.predict(X_test_pca)
print "predicting time:", round(time()-t1, 3), "s"
#Get Accuracy, Precision, and Recall 
evaluate(X_test_pca, labels_test)

print "Non-PCA Set:"
#retrain the classifier wihtout PCA 
#time the training
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
#time the prediction
t1 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time()-t1, 3), "s"

#Get Accuracy, Precision, and Recall 
evaluate(features_test, labels_test)


# Final Pipeline 
from sklearn.pipeline import Pipeline
estimators = [('MinMaxScaler', MinMaxScaler()), ('reduce_dim', PCA(n_components=1)), ('clf', \
        SVC(class_weight= 'balanced',\
        C=1, \
        gamma =.3))]
clf = Pipeline(estimators)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)