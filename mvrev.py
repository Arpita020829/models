import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer,ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df=pd.read_csv("moviereview.csv")

X=df["review"]
Y=df["sentiment"]



sw=set(ENGLISH_STOP_WORDS)
sw.discard("not")
sw.discard("no")
vect=CountVectorizer(stop_words=list(sw),ngram_range=(1,3))
punc=string.punctuation
wnet=WordNetLemmatizer()
def preproess(sent):
    sent=sent.lower()
    tokens=word_tokenize(sent)
    puc_word=[word for word in tokens if word not in punc]
    filter_word=[word for word in puc_word if word not in sw]

    temp=[]

    for word in filter_word:
        temp.append(wnet.lemmatize(word))
    return " ".join(temp)

X=X.apply(preproess)
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=.2,random_state=42)
X_train_vec=vect.fit_transform(X_train)
X_test_vec=vect.transform(X_test)

model=LogisticRegression(max_iter=1000)

model.fit(X_train_vec,y_train)

y_pred=model.predict(X_test_vec)

acuracy=accuracy_score(y_test,y_pred)
print(acuracy)

new_review= [
    "This movie was amazing and I loved it",
    "The film was boring and a waste of time",
    "Great acting and fantastic story",
    "Worst movie ever made",
    "i was not impressed by the acting in this movie",
    "such a beautiful performance by the cast of the movie"
]

new_reviews=[preproess(r) for r in new_review]
new_rev_vec=vect.transform(new_reviews)

new_pred=model.predict(new_rev_vec)
for rev,sem in zip(new_review,new_pred):
    print(rev)
    print(sem)
