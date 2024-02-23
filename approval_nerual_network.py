# Machine Learning algorithm that classifies loaners as a defualt risk or not
# Uses a neural network for training
# (machine learning practice)
# Author: John Butterfield
# Date: 2/23/2024

#import modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


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

    predictors_preprocessed = preprocessor.fit_transform(predictors)

    # split dataset into training and testing sets
    # test 30% of the data, train on 70% of the data
    predictors_train, predictors_test, target_train, target_test = train_test_split(predictors_preprocessed, target, test_size=0.3, random_state=42)

    # Define the neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(predictors_train.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # Early stopping callback
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    # Assuming predictors_train is a scipy.sparse.csr_matrix
    predictors_train = predictors_train.toarray()  # Convert to dense NumPy array
    predictors_test = predictors_test.toarray()  # Convert to dense NumPy array


    # Train the model
    history = model.fit(predictors_train, target_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping_cb])

    # Evaluate the model on the test set
    print(model.evaluate(predictors_test, target_test))

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    
if __name__ == '__main__':
    main()
