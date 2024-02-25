# Machine Learning algorithm that classifies loaners as a defualt risk or not
# Uses a neural network for training
# (machine learning practice)
# Author: John Butterfield
# Date: 2/23/2024

# import modules
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (except for errors)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.callbacks import EarlyStopping
from sklearn.base import clone
import seaborn as sns


def main():

    # working with adding experimental features to improve accuracy
    exp_feature = bool(int(input('Enter 0 for regular features or 1 for experimental features enabled: ')))

    if ( exp_feature ):
        print('Experimental feature, Income_Employment_Interaction enabled')
    else:
        print('Proceeding with default features')

    print('Loading dataset...')
    print()
    data = pd.read_csv('Applicant-details.csv')

    # pre-processing data sets includes transforming entries into valid forms a ML algorithm can use
    print('Begin pre-process of dataset...')
    print()

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

    predictors_preprocessed = preprocessor.fit_transform( predictors )

    # split dataset into training and testing sets
    # test 30% of the data, train on 70% of the data
    predictors_train, predictors_test, target_train, target_test = train_test_split( predictors_preprocessed, target, test_size=0.3, random_state=42 )

    predictors_train = predictors_train.toarray()  # Convert to dense NumPy array
    predictors_test = predictors_test.toarray()  # Convert to dense NumPy array
    
    print('Begin training models...')
    print()
            
    # Hyperparameters to search
    batch_sizes = [ 64, 32, 128, 256 ]
    epochs_list = [ 100 ]
    optimizer_options = [ 'adam' ]
    layer_size = [ 'large', 'small' ]


    # Track the best performing model
    best_accuracy = 0
    best_params = {}
    best_history = None  # To store the training history of the best model
    model_count = 1

    # Loop over each combination of hyperparameters
    for batch_size in batch_sizes:
        for epochs in epochs_list:
            for optimizer in optimizer_options:
                for size in layer_size:
                    print( f'Begin training model #{model_count}' )
                    print( f'batch_size={batch_size}, epochs={epochs}, optimizer={optimizer}, size={size}' )

                    if size == 'large':
                        # Define the 'large' model
                        model = tf.keras.Sequential([
                            tf.keras.layers.Dense(128, activation='relu', input_shape=(predictors_train.shape[1],)),
                            tf.keras.layers.Dropout(0.2),
                            tf.keras.layers.Dense(64, activation='relu'),
                            tf.keras.layers.Dense(32, activation='relu'),
                            tf.keras.layers.Dense(16, activation='relu'),
                            tf.keras.layers.Dense(1, activation='sigmoid')
                        ])
                    else:
                        # Define the 'small' model
                        model = tf.keras.Sequential([
                            tf.keras.layers.Dense(128, activation='relu', input_shape=(predictors_train.shape[1],)),
                            tf.keras.layers.Dropout(0.2),
                            tf.keras.layers.Dense(64, activation='relu'),
                            tf.keras.layers.Dense(32, activation='relu'),
                            tf.keras.layers.Dense(1, activation='sigmoid')
                        ])
                        
                        

                    # Compile the model
                    model.compile(optimizer=optimizer,
                            loss='binary_crossentropy',
                            metrics=['accuracy'])

                    # Early stopping callback
                    early_stopping_cb = tf.keras.callbacks.EarlyStopping( patience=5, restore_best_weights=True )

                    # Train the model
                    history = model.fit(predictors_train, target_train, epochs=epochs, batch_size=batch_size,
                                validation_split=0.2, callbacks=[early_stopping_cb], verbose=0)

                    # Evaluate the model on the validation set
                    val_accuracy = max(history.history['val_accuracy'])
                    print(f"Validation Accuracy: {val_accuracy:.5}")

                    # Check if this model performed better; if so, update best_accuracy and best_params
                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        best_params = {'batch_size': batch_size, 'epochs': epochs, 'optimizer': optimizer, 'layer size': size}
                        best_history = history
                        print("*** New best model found: *** ")
                    
                    print('------------------------------------')
                    print()
                    model_count += 1

    print('---------------------------------')
    print( 'Best model found: ')
    print( best_params )
    print( f'with, {best_accuracy:.5} accuracy' )
    
    # Plot training & validation accuracy values
    plt.plot(best_history.history[ 'accuracy' ])
    plt.plot(best_history.history[ 'val_accuracy' ])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(best_history.history['loss'])
    plt.plot(best_history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    
if __name__ == '__main__':
    main()
