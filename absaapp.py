from preprocess import casefolding,token,stopwords_removal,stemming
import pandas as pd
from PIL import Image
import streamlit as st
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

image=Image.open("C:/Users/win10/Downloads/—Pngtree—set of food doodle vector_5306662.png")
st.image(image)

st.title('Food Product Review Analysis Based on Aspect')

df=pd.read_csv("df_clean.csv",encoding='latin1')
size=round(len(df)*0.3)
testing_data=df[:size]
tf = TfidfVectorizer()
text_tf=tf.fit_transform(df['Review'].astype('U'))
X_train,X_test,y_train,y_test=train_test_split(text_tf,df['Aspek'],test_size=0.3,random_state=42)
clf = MultinomialNB().fit(X_train,y_train)

def training(clf,X_test,y_test):
    predicted = clf.predict(X_test)
    st.write(classification_report(y_test,predicted,zero_division=0))
    
def testing(tf,testing_data):
    testing_data=testing_data.astype({'Review':'string'})
    text_hm=tf.transform(testing_data['Review'].astype('U'))
    heh=clf.predict(text_hm)
    return heh

def preprocess(df):
    df['Review']=df['Review'].apply(casefolding)
    df['Review']=df['Review'].apply(token)
    df['Review']=df['Review'].apply(stopwords_removal)
    df['Review']=df['Review'].apply(stemming)
    return df

st.sidebar.write('Click button for training')
test3=st.sidebar.button('Training')

st.sidebar.write('Click button for testing')
test2=st.sidebar.button('Testing')

uploaded_file = st.sidebar.file_uploader("Upload Your Reviews Dataset", type=["csv"])
if uploaded_file is not None:
    uploaded_file = pd.read_csv(uploaded_file)
    btn=st.sidebar.button('Preprocess and Predict The Aspect')
    if btn:
        obj1=preprocess(uploaded_file)
        obj2=testing(tf,obj1)
        df=pd.DataFrame({'Pre-Processing Review':uploaded_file['Review'],'Aspect Prediction':obj2})
        st.write(df)

if test3:
    training(clf,X_test,y_test)

elif test2:
    ob=testing(tf,df)
    df=pd.DataFrame({'Pre-Processing Review':df['Review'],'Aspect Prediction':ob})
    st.write(df)

