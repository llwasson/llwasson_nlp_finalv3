'''
References:
https://getbootstrap.com/docs/5.1/examples/
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_pickle.html 
https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4
'''

from flask import Flask, render_template, request, redirect
import pandas as pd
import nltk
from nltk.corpus import sentiwordnet as swn
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity




app = Flask(__name__)



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/methodology/")
def methodology():
    return render_template("methodology.html") 

    
@app.route("/matches/", methods = ['POST', 'GET'])
def matches():
    df_outlook = get_outlook()
    df_matches = get_matches()
    print(df_matches)
    print(df_outlook)
    req_type = request.method
    if req_type == 'GET':  
        return render_template("matches.html", tables=[df_matches.to_html(classes='data')], titles=df_matches.columns.values)
    if request.method == 'POST':
        form_data = request.form
        return render_template('matches.html', form_data = form_data)

  
 # do pickling in jupyter notebook script?


def get_outlook():
    data_outlook = pd.read_pickle('cleaned_outlook_corpus.pkl')
    df_o = pd.DataFrame(data_outlook)
    df_o_head = df_o.head()
    df_o_head = df_o_head[['O*NET-SOC Code', 'Title', 'outlook_pred']]
    return df_o_head

def get_recommendation(top, occupations_new, scores):
    recommendation = pd.DataFrame(columns = ['JobID',  'title', 'score', 'sentiment'])
    count = 0
    for i in top:
      # recommendation.at[count, 'ID'] = resID
      recommendation.at[count, 'JobID'] = occupations_new['O*NET-SOC Code'][i]
      recommendation.at[count, 'title'] = occupations_new['title'][i]
      recommendation.at[count, 'score'] =  scores[count]
      recommendation.at[count, 'sentiment'] = occupations_new['outlook_pred'][i]
      count += 1
    return recommendation



def cos_similarity(df_res, df_occu):
    # Feature extraction
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_occupations = tfidf_vectorizer.fit_transform((df_occu['cleaned_text'])) 
    tfidf_resumes = tfidf_vectorizer.transform(df_res['Resume_str_cleaned']) 
    
    #Cosine similarity
    cos_similarity_tfidf = map(lambda x: cosine_similarity(tfidf_resumes, x),tfidf_occupations)
    
    # Convert the cosine similarities into a list
    r = list(cos_similarity_tfidf)

    # Top 10 occupational recommendations
    top = sorted(range(len(r)), key=lambda i: r[i], reverse=True)[:10]
    list_scores = [r[i][0][0] for i in top]
    data = pd.read_pickle('cleaned_outlook_corpus.pkl')
    occupations_new = pd.DataFrame(data)
    df_rec = get_recommendation(top, occupations_new, list_scores)
    return df_rec


def get_matches():
    data_resume = pd.read_pickle('cleaned_resume_corpus.pkl')
    df_r = pd.DataFrame(data_resume)
    df_r_head = df_r.head(1)
    data_occs = pd.read_pickle('cleaned_occ_corpus.pkl')
    df_occs = pd.DataFrame(data_occs)
    job_matches = cos_similarity(df_r_head, df_occs)
    return job_matches


'''
data = pd.read_pickle('cleaned_outlook_corpus.pkl')
data_frame = pd.DataFrame(data)
print(data_frame.head())
'''