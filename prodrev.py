import pandas as pd
import string
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer,ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer,ENGLISH_STOP_WORDS

df=pd.read_csv("feedback.csv")
X=df["feedback"]
Y=df["label"]
punc=string.punctuation
wnet=WordNetLemmatizer()
sw=set(ENGLISH_STOP_WORDS)
sw.discard("no")
sw.discard("not")
#vect=CountVectorizer(stop_words=list(sw),ngram_range=(1,3))
vect=TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS),ngram_range=(1,3))
def preprocess(sent):
    sent=sent.lower()
    tokens=word_tokenize(sent)
    punc_word=[word for word in tokens if word not in punc]

    temp=[]

    for word in punc_word:
        temp.append(wnet.lemmatize(word))

    return " ".join(temp)

X=X.apply(preprocess)

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=.2,random_state=42)

X_train_vecc=vect.fit_transform(X_train)
X_test_vec=vect.transform(X_test)

model=LogisticRegression(max_iter=10000)

model.fit(X_train_vecc,y_train)
y_pred=model.predict(X_test_vec)

acc=accuracy_score(y_test,y_pred)
print(acc)

tricky_reviews = [
     "This product is amazing , works perfectly",
    "Absolutely loved !! it highly recommend",
    "Great quality and fast delivery",
    "Very satisfied with the purchase",
    "Excellent service and fantastic product",
    "Superb experience will definitely buy again",
    "I am extremely happy with this product!!",
    "Very bad product completely useless",
    "I hate this item , it stopped working",
    "Terrible experience will never buy again",
    "Worst purchase I have ever made",
    "The prod was boring and a waste of time",
    "Poor quality and very disappointing",
    "Delivery was extremely late and the product was damaged",
    "Not worth the money at all",
    "i liked this product"
]

trick_rev=[preprocess(r) for r in tricky_reviews]

rev_vec=vect.transform(trick_rev)

new_pred=model.predict(rev_vec)

for rev,p in zip(tricky_reviews,new_pred):
    print(rev)
    print(p)