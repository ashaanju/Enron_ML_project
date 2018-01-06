#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# Starting with 20 features excluding email address of the person
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees','to_messages','from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

# Identification of outliers using plots is done in the file outlier_detection.py
data_dict.pop('TOTAL')

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# New features identified are as follows
# salary_to_adjusted_total_payments ie total_payments minus loan advances
# exercised_stock_option_to_stock_value
# deferred_income_to_total_payments
# stock_value_to_total_value
# restricted_stock_to_stock_value
# shared_receipt_with_poi_to_messages
# from_poi_to_messages
# to_poi_to_messages


## Function to create new feature as a ratio of two input features
def create_new_feature(feature1, feature2):
    value = 0
    if feature1 != "NaN" and feature2 != "NaN": 
	if feature2 != 0:
            value = float(feature1)/float(feature2)	
    return value

## Calculating new datapoints
for point in my_dataset.keys():
    salary = my_dataset[point]["salary"]
    loan_advances = my_dataset[point]["loan_advances"]
    total_payments = my_dataset[point]["total_payments"]  
    deferred_income = my_dataset[point]["deferred_income"]
    restricted_stock = my_dataset[point]["restricted_stock"]
    exercised_stock_options = my_dataset[point]["exercised_stock_options"]
    total_stock_value = my_dataset[point]["total_stock_value"]  
    shared_receipt_with_poi = my_dataset[point]["shared_receipt_with_poi"]  
    to_messages = my_dataset[point]["to_messages"]  
    from_messages = my_dataset[point]["from_messages"]  
    from_poi_to_this_person = my_dataset[point]["from_poi_to_this_person"]  
    from_this_person_to_poi = my_dataset[point]["from_this_person_to_poi"]  
    
    if loan_advances == 'NaN':
	loan_advances = 0
    if total_payments == 'NaN':
	total_payments = 0	    
    my_dataset[point]["deferred_income_to_total_payments"] = create_new_feature(deferred_income, total_payments)	    
    my_dataset[point]["restricted_stock_to_stock_value"] = create_new_feature(restricted_stock, total_stock_value)	    
    my_dataset[point]["exercised_stock_option_to_stock_value"] = create_new_feature(exercised_stock_options, total_stock_value)	    
    my_dataset[point]["stock_value_to_total_payments"] = create_new_feature(total_stock_value, total_payments)	    
    my_dataset[point]["shared_receipt_with_poi_to_messages"] = create_new_feature(shared_receipt_with_poi, to_messages)	    
    my_dataset[point]["to_poi_to_messages"] = create_new_feature(from_this_person_to_poi, from_messages)	    
    my_dataset[point]["from_poi_to_messages"] = create_new_feature(from_poi_to_this_person, to_messages)	    
   

### Extract features and labels from dataset for local testing

## Updated feature list
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees','to_messages','from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', "deferred_income_to_total_payments", "restricted_stock_to_stock_value", "exercised_stock_option_to_stock_value", "stock_value_to_total_payments", "shared_receipt_with_poi_to_messages", "to_poi_to_messages", "from_poi_to_messages"] 

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.metrics import classification_report

random.seed(1234)

## Defining four pipelines, one for each classifier
pipe1 = Pipeline([('scale',MinMaxScaler()),('feat', SelectKBest()),('clf', GaussianNB())])
pipe2 = Pipeline([('scale',MinMaxScaler()),('feat', SelectKBest()),('clf', DecisionTreeClassifier())])
pipe3 = Pipeline([('scale',MinMaxScaler()),('feat', SelectKBest()),('clf', LogisticRegression())])
pipe4 = Pipeline([('scale',MinMaxScaler()),('feat', SelectKBest()),('clf', KNeighborsClassifier())])


## Parameter grid for each pipeline
param_grid1 = [{'feat__k': (5,8,10,12,15,20,25)}]
param_grid2 = [{'feat__k': (5,8,10,12,15,20,25), 'clf__min_samples_leaf' : (1,2,3),'clf__min_samples_split' : (2,3,4)}]
param_grid3 = [{'feat__k': (5,8,10,12,15,20,25), 'clf__C' : (0.01,0.1,1,5,10,50)}]
param_grid4 = [{'feat__k': (5,8,10,12,15,20,25), 'clf__n_neighbors' : (1,3,5,7,9)}]

## Gridsearch on each pipeline
grid1 = GridSearchCV(pipe1, param_grid = param_grid1, scoring = "f1_weighted")
grid2 = GridSearchCV(pipe2, param_grid = param_grid2, scoring = "f1_weighted")
grid3 = GridSearchCV(pipe3, param_grid = param_grid3, scoring = "f1_weighted")
grid4 = GridSearchCV(pipe4, param_grid = param_grid4, scoring = "f1_weighted")

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

print sum(labels_test)
print "\n"
print "Results of GaussianNB"
grid1.fit(features_train, labels_train)
y1_prediction = grid1.predict( features_test )
report1 = classification_report( labels_test, y1_prediction )
print report1
print grid1.best_score_
print grid1.best_estimator_


print "\n"
print "Results of Decision tree classifier"
grid2.fit(features_train, labels_train)
y2_prediction = grid2.predict( features_test )
report2 = classification_report( labels_test, y2_prediction )
print report2
print grid2.best_score_
print grid2.best_estimator_


print "\n"
print "Results of LogisticRegression classifier"
grid3.fit(features_train, labels_train)
y3_prediction = grid3.predict( features_test )
report3 = classification_report( labels_test, y3_prediction )
print report3
print grid3.best_score_
print grid3.best_estimator_



print "\n"
print "Results of KNeighbors classifier"
grid4.fit(features_train, labels_train)
y4_prediction = grid4.predict( features_test )
report4 = classification_report( labels_test, y4_prediction )
print report4
print grid4.best_score_
print grid4.best_estimator_


print "\n"
print "Results of Best classifier"
best_pipe = Pipeline([('feat', SelectKBest(k = 12)),('clf', GaussianNB())])
best_pipe.fit(features_train, labels_train)
best_prediction = best_pipe.predict(features_test)
report_best = classification_report( labels_test, best_prediction )
print report_best
key_features= best_pipe.named_steps['feat'].get_support()
print "Selected Features"
for i in range(len(key_features)):
    if key_features[i]:
	print features_list[i+1]

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(best_pipe, my_dataset, features_list)
