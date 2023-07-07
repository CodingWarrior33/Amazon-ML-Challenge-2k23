import pandas as pd
import numpy as np
import autokeras as ak
import re
import string
from nltk.corpus import stopwords
import tensorflow as tf

train_df = pd.read_csv("train_comb.csv").dropna()
print(train_df.isna().sum())

test_df = pd.read_pickle("test_cleaned.pkl")
y_submission = np.zeros(len(test_df))
result = pd.DataFrame(columns=["PRODUCT_ID", "PRODUCT_LENGTH"])

train_df.loc[train_df['PRODUCT_LENGTH'] > 1000, 'PRODUCT_LENGTH'] = 1000

train_df.to_csv("reduced_train_data_5.csv")

model = ak.TextRegressor(overwrite=True, max_trials=1, loss='mean_absolute_error', metrics=['mae'])
model.fit(x=np.array(train_df["COMB"]), y=np.array(train_df["PRODUCT_LENGTH"]), epochs=8, validation_split=0.2,
          validation_data=None)

model.export_autokeras_model('text_regressor')

model = tf.keras.models.load_model('text_regressor')

test_df = pd.read_pickle("dataset/test_cleaned.pkl")

nltk.download('stopwords')
sw_nltk = stopwords.words('english')

def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def cleaner(text):
    if text == '':
        return ''
    text = remove_html_tags(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = [word for word in text.split(' ') if word not in sw_nltk]
    return " ".join(words)

test_df['TITLE'] = test_df['TITLE'].apply(lambda x: cleaner(x) if not pd.isna(x) else x)
test_df['DESCRIPTION'] = test_df['DESCRIPTION'].apply(lambda x: cleaner(x) if not pd.isna(x) else x)
test_df['BULLET_POINTS'] = test_df['BULLET_POINTS'].apply(lambda x: cleaner(x) if not pd.isna(x) else x)

test_df = test_df.replace(np.nan, "", regex=True)

test_df["COMB"] = test_df["TITLE"] + test_df["DESCRIPTION"] + test_df["BULLET_POINTS"]

test_df.isnull().sum()

y_submission1 = model.predict(np.array(test_df["COMB"]))
y_submission2 = model.predict(np.array(test_df["TITLE"]))

y_submission = pd.DataFrame(y_submission1)
y_submission.to_csv("comb_submission.csv", index=False)
