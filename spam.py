import pandas as pd
import nltk 
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer,ENGLISH_STOP_WORDS
import string
from sklearn.naive_bayes import MultinomialNB

df=pd.read_csv("spam.csv")
print(df.columns)
#print(df)
X=df["email"]
Y=df["label"]

wnet=WordNetLemmatizer()
sw=set(ENGLISH_STOP_WORDS)
vect=CountVectorizer(stop_words=list(sw),ngram_range=(1,2))
punc=string.punctuation
def process(sent):
    sent=sent.lower()

    tokens=word_tokenize(sent)

    filter_word=[word for word in tokens if word not in punc]
    temp=[]

    for word in filter_word:
        temp.append(wnet.lemmatize(word))

    return " ".join(temp)

X=X.apply(process)

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=.2,random_state=42)

X_train_vec=vect.fit_transform(X_train)
X_test_vec=vect.transform(X_test)
model=MultinomialNB()
model.fit(X_train_vec,y_train)

y_pres=model.predict(X_test_vec)
acc=accuracy_score(y_test,y_pres)
print(acc)

test_emails = [

# # spam-like
# "You have won a free vacation claim your prize now",
# "Earn money fast from home with this amazing opportunity",
# "Congratulations you are selected for a special reward",
# "Get cheap loans instantly apply now",
# "Limited time offer click here to win a free phone",
# "Your account is eligible for a big cash bonus",
# "Claim your free shopping voucher today",
# "Act fast to receive your exclusive reward",
# "Win exciting prizes by signing up now",
# "Free membership available for selected users"

# # ham-like
# "Please send me the updated project report",
# "The meeting has been scheduled for tomorrow",
# "Can we reschedule our discussion to Friday",
# "I have attached the files for your reference",
# "Let's finalize the presentation slides today",
# "Please review the document and give feedback",
# "The team meeting will start at 2 pm",
# "Thank you for helping with the project",
# "I will call you later to discuss the details",
# "Please confirm your attendance for the meeting"
# tricky cases
"Free parking available at the office building",
"Win the match tomorrow team practice today",
"Your order confirmation has been sent",
"Claim your booking confirmation at the counter",
"Discount available for employees in the cafeteria",
"Meeting reminder claim your attendance",
"Your bank account statement is ready",
"Exclusive invitation to attend the seminar",
"Get your course certificate from the portal",
"Limited seats available for the workshop"

]

test=[process(r) for r in test_emails]

test_vec=vect.transform(test)

new_pred=model.predict(test_vec)

for email, s in zip(test_emails,new_pred):
    print(email)
    print(s)