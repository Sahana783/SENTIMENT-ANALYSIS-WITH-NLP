COMPANY: CODTECH IT SOLUTIONS

NAME: SAHANA

INTERN ID: CT04DZ1200

DOMAIN: MACHINE LEARNING

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH

##DESCRIPTION OF THE TASK 2: SENTIMENT ANALYSIS WITH NLP

The primary objective of this code is to perform sentiment classification on customer reviews using natural language processing (NLP) techniques along with machine learning. It aims to automatically determine whether a review expresses a positive or negative sentiment. This is achieved by converting textual data into numerical format using TF-IDF (Term Frequency–Inverse Document Frequency) and feeding that 
into a Logistic Regression classifier. The model is trained to identify patterns in the text that are indicative of sentiment polarity and make accurate predictions on new, unseen reviews.

To accomplish this, several Python libraries are used. The pandas library plays a crucial role in handling and manipulating tabular data, such as the dataset of reviews. NumPy is used for general numerical computations, although its role is minimal in this particular script. For visualization, matplotlib.pyplot and seaborn are used to display the confusion matrix in a graphical format. Most importantly, scikit-learn provides essential machine learning utilities, including tools to split data (train_test_split), transform text data, train models (Logistic Regression), and evaluate them (classification_report, accuracy_score, and confusion_matrix).

The Natural Language Toolkit (nltk) is used for text preprocessing, specifically for stopword removal. Stopwords are common English words such as “is,” “the,” “an,” etc., which don’t contribute much meaning in
a sentiment context and are typically removed during preprocessing. The code first downloads the list of stopwords using nltk.download('stopwords'), and then uses stopwords.words('english') to retrieve the list for filtering.

Rather than loading an external dataset, this example defines a small in-memory dataset using a Python dictionary. It contains 10 entries with two fields: review and sentiment. Reviews are textual data, while sentiments are labeled as 1 (positive) or 0 (negative). This dataset simulates user reviews, typical of product feedback on e-commerce platforms. The dictionary is converted into a DataFrame using pandas.DataFrame, which simplifies data operations and manipulation.

Before text data can be fed into a machine learning model, it needs to be cleaned and standardized. The function clean_text is defined for this purpose. It first converts all characters in the text to lowercase to ensure consistency. It then removes HTML tags using regular expressions, as well as URLs. Additionally, punctuation and digits are stripped to retain only meaningful words. Finally, stopwords are removed 
using the predefined list from NLTK. This cleaned text is stored in a new column called clean_review in the DataFrame.

To evaluate the model’s ability to generalize to new data, the dataset is split into training and testing subsets using train_test_split. The default split ratio is 70% training and 30% testing. The features 
(X) are the cleaned review texts, and the target (y) is the sentiment label. This step ensures that the model is trained on a subset of the data and tested on unseen data to evaluate its predictive power.

Since machine learning models require numerical input, textual reviews must be transformed into numerical vectors. The code uses TfidfVectorizer to achieve this. TF-IDF stands for Term Frequency–Inverse Document Frequency and helps in identifying which words in a document are significant relative to the whole dataset. fit_transform is applied on the training set to learn the vocabulary and transform the reviews into a
TF-IDF matrix. The same vectorizer is then used to transform the test set using transform, ensuring consistency in feature space.

The classification model used in this script is LogisticRegression, a widely-used algorithm for binary classification tasks. It models the probability that a given input belongs to a particular class. In this case, the model is trained to predict whether a review is positive or negative. The fit method is used on the TF-IDF-transformed training data along with the sentiment labels. Logistic regression is chosen for its simplicity, interpretability, and effectiveness in text classification problems.

Once the model is trained, it makes predictions on the test set using the predict method. These predictions are compared against the actual sentiment labels to evaluate the model's performance. The accuracy_
score provides a basic metric indicating the percentage of correct predictions. A more detailed analysis is provided by classification_report, which includes precision, recall, and F1-score for both classes. These metrics help assess whether the model performs well across all categories or favors one over the other.

To further evaluate the classifier, a confusion matrix is generated using confusion_matrix. This matrix shows the counts of true positives, false positives, true negatives, and false negatives. For better understanding, this matrix is visualized using seaborn.heatmap, which plots the matrix with annotations. The diagonal elements represent correct classifications. A good model will have higher numbers along
the diagonal and lower values elsewhere, indicating fewer misclassifications.

The methodology demonstrated in this code has several practical applications. Businesses can use such sentiment analysis tools to monitor customer satisfaction through product reviews or social media comments. Customer support teams can prioritize responses to negative feedback flagged by such models. Marketing teams can evaluate public sentiment toward campaigns or new product launches. The model can also be 
extended to analyze sentiments in news articles, political opinions, and even movie or book reviews.

While this code demonstrates the core concept using a small dataset, the same approach can be scaled to thousands or millions of reviews. Additional improvements can be made by incorporating bigrams or trigrams into TF-IDF, tuning hyperparameters of the logistic regression model, or even replacing the model with more advanced deep learning techniques such as recurrent neural networks (RNNs) or transformer-based models like BERT. These enhancements can further improve accuracy and robustness in real-world applications.

By concluding, this task is done by using chatgpt.com and this sentiment analysis project demonstrates a full NLP pipeline: from preprocessing raw text data to transforming it with TF-IDF, training a machine learning model, evaluating its performance, and visualizing results. The modular nature of the code makes it adaptable to different types of text classification tasks. With relatively simple tools, it is possible to build a powerful and practical model for understanding and classifying human opinions expressed in text format.

##OUTPUT 

<img width="1625" height="842" alt="Image" src="https://github.com/user-attachments/assets/787b0649-6192-422e-bf44-59da0c171ff9" />
<img width="1615" height="800" alt="Image" src="https://github.com/user-attachments/assets/7e01d13a-6540-466b-9fa6-4fbb786c1666" />
