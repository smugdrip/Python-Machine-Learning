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

# class for early stopping for grid search
class KerasClassifierWithEarlyStopping(KerasClassifier):
    def __init__(self, build_fn=None, **sk_params):
        super(KerasClassifierWithEarlyStopping, self).__init__(build_fn, **sk_params)
        # Initialize the early stopping callback
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    def fit(self, X, y, **kwargs):
        # If not already included, add early stopping to callbacks
        if 'callbacks' not in kwargs:
            kwargs['callbacks'] = [self.early_stopping]
        else:
            if not any(isinstance(callback, EarlyStopping) for callback in kwargs['callbacks']):
                kwargs['callbacks'].append(self.early_stopping)
                
        return super(KerasClassifierWithEarlyStopping, self).fit(X, y, **kwargs)

def create_model( optimizer='adam', activation='relu', input_dim=None ):
    model = Sequential()
    model.add(Dense(12, input_dim=input_dim, activation=activation))
    model.add(Dense(8, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def main():

    # working with adding experimental features to improve accuracy
    exp_feature = bool(int(input('Enter 0 for regular features or 1 for experimental features enabled: ')))

    if ( exp_feature ):
        print('Experimental feature, Income_Employment_Interaction enabled')
    else:
        print('Proceeding with default features')

    batch_search = bool(int(input('Enter 0 for normal or 1 for batch search: ')))

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

    num_features = predictors_preprocessed.shape[1]
    print(f"Number of features after preprocessing: {num_features}")

    # split dataset into training and testing sets
    # test 30% of the data, train on 70% of the data
    predictors_train, predictors_test, target_train, target_test = train_test_split(predictors_preprocessed, target, test_size=0.3, random_state=42)

    # Assuming predictors_train is a scipy.sparse.csr_matrix
    predictors_train = predictors_train.toarray()  # Convert to dense NumPy array
    predictors_test = predictors_test.toarray()  # Convert to dense NumPy array

    if ( batch_search ):
        # Define the grid search parameters
        param_grid = {
            'batch_size': [ 20, 40 ],
            'epochs': [ 20, 40 ],
            'optimizer': ['Nadam'],
            'activation': ['relu'],
            'input_dim': [num_features]  # Ensure this is passed as a fixed parameter
        }

        # Define the neural network model
        model = KerasClassifier( build_fn=create_model, input_dim=num_features, verbose=0 )

        # Grid search
        grid = GridSearchCV( estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose=0 )
        grid_result = grid.fit( predictors_train, target_train )

        # Summarize results
        print( "Best: %f using %s" % ( grid_result.best_score_, grid_result.best_params_) )
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
    
    else:
            
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
