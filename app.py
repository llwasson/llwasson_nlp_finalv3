'''
References:
https://getbootstrap.com/docs/5.1/examples/
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_pickle.html 
https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4
https://stackabuse.com/flask-form-validation-with-flask-wtf/
https://stackabuse.com/deploying-a-flask-application-to-heroku/
https://careerkarma.com/blog/css-tables/
'''

from flask import Flask, render_template, request, redirect
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField
from wtforms.validators import ValidationError, DataRequired, NumberRange
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



class OccForm(FlaskForm):
    resID = IntegerField(label=("Enter Resume ID"), validators=[DataRequired()])
    topN = IntegerField(label=("Enter number of jobs to view (1-20)"), validators=[DataRequired(), NumberRange(min=1, max=20, message="Value must be between 1 and 20")])
    submit = SubmitField(label=('Submit'))


app = Flask(__name__)
app.config['SECRET_KEY'] = 'FinalProjectKey'




@app.route("/", methods=['GET', 'POST'])
def home():
    form = OccForm()
    if form.validate_on_submit() and request.method == 'POST':
        df_matches = get_matches(form.resID.data, form.topN.data)
        if df_matches.shape[0] == 0:
            return render_template("noresults.html")
        else:
            return render_template("matches.html", tables=[df_matches.to_html(classes='data', index=False)], titles=df_matches.columns.values, name=form.resID.data, top=form.topN.data)
    return render_template("index.html", form=form)

@app.route("/methodology/")
def methodology():
    return render_template("methodology.html") 



def get_recommendation(top, occupation_match, scores):
    recommendation = pd.DataFrame(columns = ['Occupation', 'Outlook'])
    count = 0
    print(top)
    print(occupation_match)
    for i in top:
      # recommendation.at[count, 'ID'] = resID
      recommendation.at[count, 'Occupation'] = occupation_match['Title'][i]
      recommendation.at[count, 'Outlook'] = occupation_match['outlook_pred'][i]
      count += 1
    return recommendation



def cos_similarity(df_res, df_occu, topX):
    # Feature extraction
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_occupations = tfidf_vectorizer.fit_transform((df_occu['cleaned_text'])) 
    tfidf_resumes = tfidf_vectorizer.transform(df_res['Resume_str_cleaned']) 
    
    #Cosine similarity
    cos_similarity_tfidf = map(lambda x: cosine_similarity(tfidf_resumes, x),tfidf_occupations)
    
    # Convert the cosine similarities into a list
    r = list(cos_similarity_tfidf)

    # Top 10 occupational recommendations
    top = sorted(range(len(r)), key=lambda i: r[i], reverse=True)[:int(topX)]
    list_scores = [r[i][0][0] for i in top]
    data = pd.read_pickle('cleaned_outlook_corpus.pkl')
    occupations_new = pd.DataFrame(data)
    df_rec = get_recommendation(top, occupations_new, list_scores)
    return df_rec


def get_matches(resumeID, topNoccs):
    data_resume = pd.read_pickle('cleaned_resume_corpus.pkl')
    df_r = pd.DataFrame(data_resume)
    print(type(resumeID))
    print(df_r.dtypes)
    print(df_r.head())
    df_resume = df_r[df_r['ID']==int(resumeID)]
    print(df_resume)
    data_occs = pd.read_pickle('cleaned_outlook_corpus.pkl')
    df_occs = pd.DataFrame(data_occs)
    print(df_occs)
    if df_resume.shape[0] == 0:
        return df_resume
    else: 
        job_matches = cos_similarity(df_resume, df_occs, topNoccs)
        return job_matches



