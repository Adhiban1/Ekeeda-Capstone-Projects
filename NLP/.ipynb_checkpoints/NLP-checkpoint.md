# Spam Mails Dataset [Kaggle]
[Data Set from Kaggle](https://www.kaggle.com/datasets/venky73/spam-mails-dataset)


```python
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
```


```python
df = pd.read_csv('spam_ham_dataset.csv', encoding='latin-1')
print('Shape:', df.shape)
df.head()
```

    Shape: (5171, 4)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>label</th>
      <th>text</th>
      <th>label_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>605</td>
      <td>ham</td>
      <td>Subject: enron methanol ; meter # : 988291\r\n...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2349</td>
      <td>ham</td>
      <td>Subject: hpl nom for january 9 , 2001\r\n( see...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3624</td>
      <td>ham</td>
      <td>Subject: neon retreat\r\nho ho ho , we ' re ar...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4685</td>
      <td>spam</td>
      <td>Subject: photoshop , windows , office . cheap ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2030</td>
      <td>ham</td>
      <td>Subject: re : indian springs\r\nthis deal is t...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



> We have to take only `label` and `text` from `df`


```python
df = df[['label', 'text']]
df.rename(columns={'text': 'message'}, inplace=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Subject: enron methanol ; meter # : 988291\r\n...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Subject: hpl nom for january 9 , 2001\r\n( see...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ham</td>
      <td>Subject: neon retreat\r\nho ho ho , we ' re ar...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>spam</td>
      <td>Subject: photoshop , windows , office . cheap ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Subject: re : indian springs\r\nthis deal is t...</td>
    </tr>
  </tbody>
</table>
</div>




```python
nltk.download('stopwords')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /home/adhiban/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.





    True



- `SnowballStemmer` is used to reduce words to their base form, also known as the root form.

- `stop words` are words that are commonly used in a language and do not carry much meaning or significance. Examples of stop words include “the”, “and”, “a”, “an”, etc.


```python
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
```


```python
nltk.download('punkt')
```

    [nltk_data] Downloading package punkt to /home/adhiban/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!





    True




```python
def clean_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if not word in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    cleaned_text = " ".join(tokens)
    return cleaned_text
```


```python
df['cleaned_message'] = df['message'].apply(clean_text)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>message</th>
      <th>cleaned_message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Subject: enron methanol ; meter # : 988291\r\n...</td>
      <td>subject enron methanol meter follow note gave ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Subject: hpl nom for january 9 , 2001\r\n( see...</td>
      <td>subject hpl nom januari see attach file hplnol...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ham</td>
      <td>Subject: neon retreat\r\nho ho ho , we ' re ar...</td>
      <td>subject neon retreat ho ho ho around wonder ti...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>spam</td>
      <td>Subject: photoshop , windows , office . cheap ...</td>
      <td>subject photoshop window offic cheap main tren...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Subject: re : indian springs\r\nthis deal is t...</td>
      <td>subject indian spring deal book teco pvr reven...</td>
    </tr>
  </tbody>
</table>
</div>




```python
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(df['cleaned_message'].values)
```


```python
classifier = MultinomialNB()
targets = df['label'].values
classifier.fit(counts, targets)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>MultinomialNB()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">MultinomialNB</label><div class="sk-toggleable__content"><pre>MultinomialNB()</pre></div></div></div></div></div>




```python
examples = ['Free smartphones', "I'm going to attend the Linux users group tomorrow."]
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
print(predictions)
```

    ['spam' 'ham']

