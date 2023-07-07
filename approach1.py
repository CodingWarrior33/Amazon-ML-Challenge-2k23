import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense

tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

def output_regression(cluster_train_path):
    model = Sequential()
    model.add(Dense(1, input_dim=X.shape[1]))
    model.compile(optimizer='adam', loss='mean_squared_error')

    df = pd.read_csv(cluster_train_path)

    attr1 = df["TITLE_DES"]
    attr2 = df["TITLE_BUL"]

    Y = df["PRODUCT_LENGTH"]

    attr1_tfidf = tfidf_vectorizer.fit_transform(attr1)
    attr2_tfidf = tfidf_vectorizer.transform(attr2)

    max_len = 128

    # Extracting the vectors
    attr1_vectors = attr1_tfidf.toarray()
    attr2_vectors = attr2_tfidf.toarray()

    if len(attr1_vectors) <= 128:
        attr1_vectors = pad_sequences(attr1_vectors, maxlen=max_len, padding='post')

    if len(attr2_vectors) <= 128:
        attr2_vectors = pad_sequences(attr2_vectors, maxlen=max_len, padding='post')

    attr1_vectors = np.array(attr1_vectors)
    attr2_vectors = np.array(attr2_vectors)

    X = np.column_stack((attr1_vectors, attr2_vectors))

    model.fit(X, Y, epochs=10)

    return model

li = []
y_submission = np.zeros(len(test_df.get_group(-1)))
result = pd.DataFrame(columns=["PRODUCT_ID", "PRODUCT_LENGTH"])

for group in test_df.groups:
    if group == -1:
        y_submission = np.zeros(len(test_df.get_group(-1)))
        li.append(y_submission)
        attr3 = test_df.get_group(-1)["PRODUCT_ID"]
    else:
        df = test_df.get_group(group)
        attr1 = df["TITLE_DES"]
        attr2 = df["TITLE_BUL"]
        attr3 = df["PRODUCT_ID"]
        
        attr1_tfidf = tfidf_vectorizer.transform(attr1)  # Use transform instead of fit_transform
        attr2_tfidf = tfidf_vectorizer.transform(attr2)  # Use transform instead of fit_transform

        max_len = 128

        # Extracting the vectors
        attr1_vectors = attr1_tfidf.toarray()
        attr2_vectors = attr2_tfidf.toarray()

        if len(attr1_vectors) > 128:  # Use greater than or equal to (>=) instead of greater than (>)
            attr1_vectors = attr1_vectors[:128]
        else:
            attr1_vectors = pad_sequences(attr1_vectors, maxlen=max_len, padding='post')

        if len(attr2_vectors) > 128:  # Use greater than or equal to (>=) instead of greater than (>)
            attr2_vectors = attr2_vectors[:128]
        else:
            attr2_vectors = pad_sequences(attr2_vectors, maxlen=max_len, padding='post')

        attr1_vectors = np.array(attr1_vectors)
        attr2_vectors = np.array(attr2_vectors)

        X_test = np.column_stack((attr1_vectors, attr2_vectors))

        url = "train_" + str(group) + ".csv"  # Simplify URL assignment using string concatenation

        model = output_regression(url)
        y_submission = model.predict(X_test)
        li.append(y_submission)
    
    new_df = pd.DataFrame({"PRODUCT_ID": attr3, "PRODUCT_LENGTH": y_submission})
    result = pd.concat([result, new_df])