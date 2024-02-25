# Machine Learning algorithm that classifies loaners as a defualt risk or not
# Uses the RandomForestClassifier for training
# (machine learning practice)
# Author: John Butterfield
# Date: 2/23/2024

#import modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


def main():

    # working with adding experimental features to improve accuracy
    exp_feature = bool(int(input('Enter 0 for regular features or 1 for experimental features enabled: ')))

    if ( exp_feature ):
        print('Experimental feature, Income_Employment_Interaction enabled')
    else:
        print('Proceeding with default features')

    print('Loading dataset...')
    data = pd.read_csv('Applicant-details.csv')

    # pre-processing data sets includes transforming entries into valid forms a ML algorithm can use
    print('Begin pre-process of dataset...')

    if ( exp_feature ):
        # Applying a scaled exponential transformation
        data['Exp_Scaled_Years_in_Current_Employment'] = np.exp(data['Years_in_Current_Employment'] / 10) - 1

        # Interaction feature using the scaled exponential transformation
        data['Income_ExpEmployment_Interaction'] = data['Annual_Income'] * data['Exp_Scaled_Years_in_Current_Employment']


    # Create predictors, which contains all features from the dataset that will be used in predicting
    # the target variable.
    predictors = data.drop(['Applicant_ID', 'Loan_Default_Risk'], axis=1)

    # Create target, which is a list of all the target variables in the data set
    target = data['Loan_Default_Risk']

    # seperate the categorical and numerical variables from the dataset into different lists
    categorical = predictors.select_dtypes( include=[ 'object', 'bool' ] ).columns
    numerical = predictors.select_dtypes( include=[ 'int64', 'float64' ] ).columns

    # Create the preprocessing pipeline for numerical data
    numerical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='mean')), # replace empty data entires with the mean value
            ('scaler', StandardScaler()) # standardize the numerical data by converting them to unit variance
        ]
    ) 
    
    # Create the preprocessing pipeline for categorical data
    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')), # replace empty data entires with the most frequent value
            ('onehot', OneHotEncoder(handle_unknown='ignore')) # convert categorical data into binary, ignoring unknown values
        ]
    ) 

    # Combine numerical and categorical pipelines into one pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical),
            ('cat', categorical_transformer, categorical)
        ]
    )
    
    
    print('Begin training model...')

    # split dataset into training and testing sets
    # test 30% of the data, train on 70% of the data
    predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, target, test_size=0.3, random_state=42)

    # Create a pipeline that combines the preprocessor with the classifier
    model = Pipeline( 
                steps=[
                    ( 'preprocessor', preprocessor ),   
                    ( 'classifier', RandomForestClassifier() )  # use a popular classifier
                ]
            )

    # Train the model using the RandomForestClassifier
    model.fit( predictors_train, target_train )

    # Predict on test data
    target_pred = model.predict( predictors_test )

    # Evaluate the model
    print(f"Accuracy: {accuracy_score( target_test, target_pred )} ")
    print( classification_report( target_test, target_pred))

    show_matrix = bool(int(input('Enter 1 to see the matrix or 0 to skip: ')))

    if ( show_matrix ):
        cm = confusion_matrix( target_test, target_pred )

        sns.heatmap( cm, annot=True, fmt='d', cmap='Blues' )
        plt.xlabel('Actual Labels')
        plt.ylabel('Predicted Labels')
        plt.show()
    
    show_roc_curve = bool(int(input('Enter 1 to see the ROC curve or 0 to skip: ')))

    if ( show_roc_curve ):
        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(target_test, model.predict_proba(predictors_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        # Plotting
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
    
    show_precision_curve = bool(int(input('Enter 1 to see the Precision curve or 0 to skip: ')))

    if ( show_precision_curve ):
        y_score = model.predict_proba(predictors_test)[:, 1]
        average_precision = average_precision_score(target_test, y_score)

        precision, recall, _ = precision_recall_curve(target_test, y_score)

        # Plotting
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'2-class Precision-Recall curve: AP={average_precision:0.2f}')
        plt.show()

if __name__ == '__main__':
    main()
