#  Automated FakeNews Detector #

### Description ###
Machine learning models can successfully detect misleading information by extracting,
analyzing this information using natural language processing algorithms. With data
mining, we developed several machine learning techniques to read anomalies that
detects fake news published on social media. In addition to a classifier, we 
created an app for visualizing the classification results.



### Libraries
The python library includes: numpy, pandas, scikit learn, matplotlib, seaborn, nltk, XGBoost, Tensorflow, Keras, Streamlit.
Please make sure you have all these libraries installed.
- All codes are written in Jupyter Notebook

### Dataset

 * The Fake and Real news dataset 44689 (cleaned from original data on Kaggle).
 * This data contains articles on different topics, but most of the articles are in on politics.
 * We have two CSV files, namely, "True.csv,” which has 21,211 and "Fake.csv, 23478 articles”.Both files have labels, article titles,
content, subject, and the period published.
 
### Running Instructions

- Collected data is cleaned by reading the CSV hives into the panda's data frames.
- Create a Bag-of-word Model where occurrence of words is captured at this point without concern for grammar.
- Transform each document to a vector representing the documents' available words.
- Create a dictionary with words in title column, while each word representing a feature.
- Use the transform() method of the count vectorizer to convert the documents in the dictionary to count vectors, non-zero values and their index are stored in a sparse matrix representation.
- Visualize the count vectors as columns with actual feature names by converting this matrix into a data frame.
- To minimize the number of features in the count vector, the max_features parameter is set to 1000.
- Sklearn.feature_extraction.text, which provides a list of pre-defined stop words in English is used to remove the stop words.
- Tokenize sentence into words and convert all words to lowercase and remove punctuations.
- The same techniques and codes are applied to the text attribute of our dataset.
- Perform feature selection on the combined attribute ‘title’ and ‘text’ to get a score of how much each feature contributes to the data.
- Use the sklearn implementation of TF-IDF to vectorize and encode the data.
- Saved the vectorizer in a pickle format which converts the python object hierarchy into a bye stream.
- Split data into train and test sets using the train_test split function using the sklearn library.
- Construct construct the training algorithm.
- Fit the trained data, x_train, which is the data attributed to concatenated title and text attributes while y_train is the class.
- The x_test attribute will be used to validate the models.
- Classification is done using algorithms, XGBoost, Linear Support Vector Machine (LSVM), Linear Regression (LR), K-Nearest Neighbor (KNN), Multinomial Naïve Bayes (NB), Decision tree (DT), Support Vector Machine (SVM) and Long Short Term Memory (LSTM).
- Build a web application using Python libraries, pandas, streamlit, and pickle.
- Design an app interface using a form to take the input title, content, and a submission button.
- The submission button initiates the prediction by combining the title and content and getting the vectorized form of the new input.
- Prediction data input using the trained model to check if news is fake or not.
- In our implementation, we used XGBoost model as it gave the best result.
