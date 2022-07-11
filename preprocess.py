import streamlit as st
import re
import string
from sklearn.pipeline import Pipeline
import numpy as np
seed =1234
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def casefolding(comment):
    comment = comment.lower()
    comment = comment.strip(" ")
    comment = re.sub(r'[?|$|!ð.\n|_:")(-+,]','',comment)
    return comment
def token (comments):
    nstr = comments.split(' ')
    dat=[]
    a=-1
    for hu in nstr:
        a=a+1
    if hu == '':
        dat.append(a)
    p=0
    b=0
    for q in dat:
        b=q-p
        del nstr[b]
        p=p+1
    return nstr
def stopwords_removal(comments):
    filtering = stopwords.words('indonesian','english')
    x=[]
    data=[]
    def myfunc(x):
        if x in filtering:
            return False
        else:
            return True
    fit = filter(myfunc,comments)
    for x in fit:
        data.append(x)
    return data
def stopwords_removal(comments):
    filtering = stopwords.words('indonesian','english')
    x=[]
    data=[]
    def myfunc(x):
        if x in filtering:
            return False
        else:
            return True
    fit = filter(myfunc,comments)
    for x in fit:
        data.append(x)
    return data
def stemming(comments):
    factory=StemmerFactory()
    stemmer = factory.create_stemmer()
    do=[]
    for w in comments:
        dt=stemmer.stem(w)
        do.append(dt)
    d_clean=[]
    d_clean=" ".join(do)
    return d_clean