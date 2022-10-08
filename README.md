# FakeNews_Net
Automated Detection of Fake news using Machine Learning Algorithms
We import the necessary libraries for analysis, EDA, visualization, and machine learning classification
We import our dataset.
The collected data is cleaned by reading the CSV hives into the panda's data frames.
We create a Bag-of-word Model where occurrence of words is captured at this point without concern for grammar.
We transform each document to a vector representing the documents' available words.
We create a dictionary with words in title column, while each word representing a feature.
We use the transform() method of the count vectorizer to convert the documents in the dictionary to count vectors.
Non-zero values and their index are stored in a sparse matrix representation.
We can visualize the count vectors as columns with actual feature names by converting this matrix into a data frame.
To minimize the number of features in the count vector, the max_features parameter is set to 1000.
Sklearn.feature_extraction.text, which provides a list of pre-defined stop words in English are usd to remove the stop words.
We tokenize our sentence into words and convert all words to lowercase.
The same techniques and codes are applied to the text attribute of our dataset.
We performed feature selection on the combined attribute ‘title’ and ‘text’ to get a score of how much each feature contributes to the data.
We use the sklearn implementation of TF-IDF to vectorize and encode our data.
Saved the vectorizer in a pickle format which converts the python object hierarchy into a bye stream.
Data is split into train and test sets using the train_test split function from the sklearn library.
Python codes are used to construct our training algorithm in Jupyter Notebook.
We fit the trained data, x_train, which is the data attributed to concatenated title and text attributes while y_train is the class.
The x_test attribute is used to validate our models.
Classification is done using algorithms like, XGBoost, Linear Support Vector Machine (LSVM), Linear Regression (LR), K-Nearest Neighbor (KNN), Multinomial Naïve Bayes (NB), Decision tree (DT), Support Vector Machine (SVM) and Long Short Term Memory.
We built a web application using Python, import pandas, streamlit, and pickle libraries environment on Jupyter.
We design an app interface using a form to take the input title, content, and a submission button.
The submission button initiates the prediction by combining the title and content and getting the vectorized form of the new input.
Lastly, the prediction of the input using the trained model to check if it is fake or now.
In our case we used XGBoost model as it gave the best result.
