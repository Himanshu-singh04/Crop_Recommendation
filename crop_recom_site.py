subprocess.call([sys.executable, '-m', 'pip', 'install', 'matplotlib'])
# standard imports
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns

# importing models
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title('Crop Recommendation System')
st.subheader('Crop Recommendation System using Machine Learning')

tab1, tab2 = st.tabs(['General Data Exploration', 'Crop Prediction'])
with tab1:
    upload = st.file_uploader('Upload your file', type=['csv'])
    if upload is not None:
        dataset = pd.read_csv(upload)

        if st.checkbox('Display Dataset'):
            if st.button('Head'):
                st.write(dataset.head())

            if st.button('Tail'):
                st.write(dataset.tail())

        if st.checkbox('Display Data Types'):
            st.text('Data Types')
            st.write(dataset.dtypes)

        if st.checkbox('Display Dataset Shape'):
            dataset_shape = st.radio(
                'What shape do you want to see?', ('Rows', 'Columns'))

            if dataset_shape == 'Rows':
                st.text('Showing Rows')
                st.write(dataset.shape[0])

            if dataset_shape == 'Columns':
                st.text('Showing Columns')
                st.write(dataset.shape[1])

        if st.checkbox('Display Summary'):
            st.write(dataset.describe())

        if st.checkbox('Display Null Values'):
            st.write(dataset.isnull().sum())

with tab2:
    crop = pd.read_csv(
        'C:\\Users\\Himanshu Singh\\OneDrive\\Desktop\\CODES\\Data_Analytics\\DSA_Lab\\Crop_Recommendation\\Crop_recommendation.csv')
    if st.checkbox('Display Dataset'):
        if st.button('Head'):
            st.write(crop.head())

        if st.button('Tail'):
            st.write(crop.tail())

    if st.checkbox('Display Data Types'):
        st.text('Data Types')
        st.write(crop.dtypes)

    if st.checkbox('Data Info'):
        st.write(crop.info())

    if st.checkbox('Display Dataset Shape'):
        dataset_shape = st.radio(
            'What shape do you want to see?', ('Rows', 'Columns'))

        if dataset_shape == 'Rows':
            st.text('Showing Rows')
            st.write(crop.shape[0])

        if dataset_shape == 'Columns':
            st.text('Showing Columns')
            st.write(crop.shape[1])

    if st.checkbox('Display all possible crops'):
        st.write(crop['label'].unique())

    normalized = preprocessing.normalize(
        crop[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]])
    crop[["N", "P", "K", "temperature", "humidity",
          "ph", "rainfall"]] = pd.DataFrame(normalized)
    correlation = crop.corr()
    if st.checkbox('Display Correlation Matrix'):
        heat_map = sns.heatmap(correlation, xticklabels=correlation.columns,
                               yticklabels=correlation.columns, annot=True)
        st.pyplot(heat_map.figure)

    if st.checkbox('Display Outliers'):
        dis = sns.boxplot(crop)
        plt.xlabel('Parameters')
        plt.ylabel('Values')
        plt.title('Boxplot for Crop Dataset')
        st.pyplot(dis.figure)

    if st.checkbox('Display Distribution'):
        feature_col = np.array(crop.drop(columns=['label']).columns)
        for col in feature_col:
            all = plt.figure()
            crop[col].hist()
            plt.title(col)
            st.pyplot(all.figure)

    x = crop.drop('label', axis=1)
    y = crop['label']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=1, test_size=0.25)

    model_name = []
    accuracy_of_model = []
    features = ['N', 'P', 'K', 'temperature',
                'humidity', 'ph', 'rainfall', 'label']

    # Multi Linear Regression Model
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Create a LinearRegression object
    multiLinearRegModel = LinearRegression()

    # Train the model
    multiLinearRegModel.fit(x_train, y_train_encoded)

    # Predict the labels for the testing dataset
    predicted_labels = multiLinearRegModel.predict(x_test)

    # Calculate the Mean Squared Error (MSE)
    mse = mean_squared_error(y_test_encoded, predicted_labels)

    # Calculate R-squared
    r_squared = r2_score(y_test_encoded, predicted_labels)

    # Calculate VIF (Variance Inflation Factor)
    vif = 1 / (1 - r_squared)

    model_name.append("Multi Linear Regression")
    accuracy_of_model.append(mse)

    # Logistic Regression Model
    logisticRegModel = LogisticRegression(
        solver='saga', random_state=0, penalty='l2', tol=0.08, class_weight='balanced')

    # training our model
    logisticRegModel.fit(x_train, y_train)

    # finding predicted labels of testing dataset
    predicted_labels = logisticRegModel.predict(x_test)

    # calculating accuracy score of our predicted labels
    score_1 = metrics.accuracy_score(y_test, predicted_labels)

    model_name.append("Logistic Regression")
    accuracy_of_model.append(score_1*100)

    # Decision Trees Model
    decisionTreeModel = DecisionTreeClassifier(
        criterion="gini", random_state=0, max_depth=10)

    # Training model
    decisionTreeModel.fit(x_train, y_train)

    # Predicting our test labels
    predicted_labels = decisionTreeModel.predict(x_test)

    # calculating accuracy score of our predicted labels
    score_2 = metrics.accuracy_score(y_test, predicted_labels)

    model_name.append("Decision Trees")
    accuracy_of_model.append(score_2*100)

    # Random Forest Model
    randomForestModel = RandomForestClassifier(
        n_estimators=20, random_state=2, criterion="gini", max_depth=9)

    # Training model
    randomForestModel.fit(x_train, y_train)

    # Predicting label of testing dataset.
    predicted_labels = randomForestModel.predict(x_test)

    # calcualting accuracy score of RF model
    score_3 = metrics.accuracy_score(y_test, predicted_labels)

    model_name.append("Random Forest")
    accuracy_of_model.append(score_3*100)

    # AdaBoost Model
    clf = AdaBoostClassifier()
    clf.fit(x_train, y_train)

    y_predict = clf.predict(x_test)

    score_4 = metrics.accuracy_score(y_test, y_predict)

    model_name.append("AdaBoost")
    accuracy_of_model.append(score_4*100)

    # AdaBoost wirh Decision Tree (Base)

    model_boost = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=9), n_estimators=6)
    model_boost.fit(x_train.values, y_train.values)

    y_predict = model_boost.predict(x_test.values)

    score_5 = metrics.accuracy_score(y_test, y_predict)
    predicted_labels = model_boost.predict(x_test)

    model_name.append("AdaBoost with Decision Tree(Base)")
    accuracy_of_model.append(score_5*100)

    # Naive Bayes Model
    gaussianNaiveBayesModel = GaussianNB(var_smoothing=0.09)

    # training our model
    gaussianNaiveBayesModel.fit(x_train, y_train)

    # predicting labels of testing set
    predicted_labels = gaussianNaiveBayesModel.predict(x_test)

    # calculating score of our testing set
    score_6 = metrics.accuracy_score(y_test, predicted_labels)

    model_name.append("Naive Bayes")
    accuracy_of_model.append(score_6*100)

    # SVM Model
    svmClassifier = SVC(kernel='poly', degree=2, tol=0.001, random_state=0)

    # training our model
    svmClassifier.fit(x_train, y_train)

    # predicting labels of testing set
    predicted_labels = svmClassifier.predict(x_test)

    # accuaray of our model on testing set
    score_7 = metrics.accuracy_score(y_test, predicted_labels)

    model_name.append("SVM")
    accuracy_of_model.append(score_7*100)

    # Neural Network
    mlpClassifier = MLPClassifier(hidden_layer_sizes=200, activation='relu',
                                  batch_size=5000, random_state=0, tol=0.0005, max_iter=500)

    # training our model
    mlpClassifier.fit(x_train, y_train)

    # predicting labels of testing set
    predicted_labels = mlpClassifier.predict(x_test)

    # accuaray of our model on testing set
    score_8 = metrics.accuracy_score(y_test, predicted_labels)

    model_name.append("Neural Network")
    accuracy_of_model.append(score_8*100)

    if st.checkbox('Compare Model Accuracy'):
        fig = plt.figure(figsize=(20, 5))
        # plotting accuracy of different model in same graph
        plt.title('Accuracy Comparison of Different Models')
        plt.ylabel('Algorithm')
        plt.xlabel('Accuracy in %')
        plt.xticks(range(0, 101, 10))

        compare = sns.barplot(y=model_name, x=accuracy_of_model)
        st.pyplot(compare.figure)
        
    if st.checkbox('Display Model Accuracy'):
        st.write('Model Accuracy on test data:', score_3*100)

    nitrogen = st.number_input('Nitrogen Content', min_value=0, max_value=200)
    phosphorous = st.number_input(
        'Phosphorous Content', min_value=0, max_value=200)
    potassium = st.number_input(
        'Potassium Content', min_value=0, max_value=200)
    temp = st.number_input('Temperature in Celsius', min_value=0, max_value=50)
    humidity = st.number_input('Relative Humidity', min_value=0, max_value=100)
    pH = st.number_input('pH', min_value=0, max_value=14)
    rainfall = st.number_input('Rainfall in mm', min_value=0, max_value=1000)

    new = [[nitrogen, phosphorous, potassium, temp, humidity, pH, rainfall]]
    pred = randomForestModel.predict(new)

    if st.button('Predict Crop'):
        st.write('Predicted Crop: ', pred[0].upper())
