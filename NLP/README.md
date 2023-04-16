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

```python
examples = ['Free smartphones', "I'm going to attend the Linux users group tomorrow."]
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
print(predictions)
```

    ['spam' 'ham']

