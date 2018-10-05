from flask import Flask, render_template, request, jsonify, render_template, url_for
import nltk
import pickle
import json
import string
import spacy
import re

nlp = spacy.load('en_core_web_sm')

nltk.download('wordnet')
nltk.download('vader_lexicon')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

tone_clf=pickle.load(open('training_models/tone/tone_clf.joblib.pkl',"rb"), encoding = "latin1")
polite_clf=pickle.load(open('training_models/politeness/classifier.joblib.pkl',"rb"), encoding = "latin1")
tf_tone=pickle.load(open('training_models/tone/vectorizer.joblib.pkl',"rb"), encoding = "latin1")
tf_polite=pickle.load(open('training_models/politeness/vectorizer.joblib.pkl', "rb"), encoding = "latin1")


@app.route('/<string:page_name>/')
def render_static(page_name):
    url_for('static', filename = 'Chart.js')
    return render_template('%s.html' % page_name)

@app.route('/postmethod', methods = ['POST'])
def get_post_email_data():
    jsdata = request.form['data']
    noisefreedata = removenoise(jsdata)
    word_count_length = word_count(noisefreedata)
    tokens = nltk.tokenize.TreebankWordTokenizer()
    tokenlist = tokens.tokenize(noisefreedata)
    resList = lemmatizeText(tokenlist)
    if resList==[]:
        return json.dumps({})

    sentences= createSentenceList(noisefreedata)

    processedData=getPunctFreeString(resList)

    sid=SentimentIntensityAnalyzer()
    scores=sid.polarity_scores(processedData)

    tone_res,tone_score=tone_clf.predict(tf_tone.transform([processedData])),tone_clf.predict_proba(tf_tone.transform([processedData]))
    polite_res, polite_score=polite_clf.predict(tf_polite.transform([processedData])), polite_clf.predict_proba(tf_polite.transform([processedData]))
    

    sentenceScores=getSentenceScores(sentences)
    sentence_count = len(sentences)
    errors = getBadSentences(sentenceScores, sentences)


    complex_words_length, syllable_count, complexwordslist = getComplexWords(resList)

    #based on this: https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
    if (sentence_count == 0 or word_count_length == 0):
        reading_level = "Not Available"
    else:
        #cast as float for python 2
        reading_level = round(100 - (206.835 - 1.015*(word_count_length/sentence_count)-84.6*(syllable_count/word_count_length)))
        if reading_level > 100:
            reading_level = 100
        if reading_level < 0:
            reading_level = 0


    scores['word_count']=word_count_length
    scores['sentence_count'] = sentence_count
    scores['syllable_count'] = syllable_count
    scores['complex_words']=complex_words_length
    scores['politeness']={'polite':round(polite_score[0, 0], 4), "rude": round(polite_score[0, 1], 4)}

    scores['tone']={'anger':round(tone_score[0,0],4),'fear':round(tone_score[0,1],4),'joy':round(tone_score[0,2],4),'love':round(tone_score[0,3],4),'sadness':round(tone_score[0,4],4),'surprise':round(tone_score[0,5],4)}
    scores['errors']=errors
    scores['complex_list'] = complexwordslist
    scores['complexity'] = reading_level
    overall_score = getOverallScore(scores)
    scores['overall_score'] = overall_score

    return json.dumps(scores)

def removenoise(input):
    #gets rid of html tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', input)
    l=cleantext.split()
    res=[]
    for string in l:
        res+=string.split('<')
    res1=[]
    for string in res:
        res1+=string.split('>')
    res2=[]
    for string in res1:
        res2+=string.split('&')
    noise=['div','/div','br','nbsp;', 'mark', '/mark', 'style="background-color:#FBBBB9;"', 'style="background-color:#FFFFC2;"', 'style="background-color:#C2DFFF;"']
    result=[x for x in res2 if x not in noise and x]
    finalstr=''
    for string in result:
        finalstr+=string+' '
    finalstr = finalstr[:len(finalstr) - 1]
    return finalstr

def lemmatizeText(tokenlist):
    stemmer=nltk.stem.WordNetLemmatizer()
    for token in tokenlist:
        token = stemmer.lemmatize(token)
    return tokenlist

def word_count(str_text):
    text = str_text.split()
    length = 0
    punctuation = ['.', '!', '?', ',', ';', ':', '"', '-', '_', '+', '=', '&', '<', '>']
    punctuation2 = ["'"]
    for token in text:
        if token not in punctuation and token not in punctuation2:
            length += 1
    return length   


def createSentenceList(text):
    sentences=[]
    currSentence=''
    punct=['.','!','?']
    i=0
    if text[len(text) - 1] not in punct:
        text+= "."
    while i<len(text):
        if text[i] not in punct:
            currSentence+=text[i]
            i+=1
        else:
            sentences.append(currSentence + text[i])
            i+=1
            while i<len(text) and text[i] in punct:
                i+=1
            currSentence=''
    return sentences

def getSentenceScores(sentences):
    sid=SentimentIntensityAnalyzer()
    sentenceScores=[]
    for sentence in sentences:
        sentence_length=word_count(sentence)
        complex_words, syl, complexwordslist=getComplexWords(sentence.split())
        scores=sid.polarity_scores(sentence)
        tone_res,tone_score=tone_clf.predict(tf_tone.transform([sentence])),tone_clf.predict_proba(tf_tone.transform([sentence]))
        polite_res, polite_score=polite_clf.predict(tf_polite.transform([sentence])), polite_clf.predict_proba(tf_polite.transform([sentence]))
        scores['word_count']=sentence_length
        scores['complex_words']=complex_words
        scores['politeness']={'polite':round(polite_score[0, 0], 4), "rude": round(polite_score[0, 1], 4)}
        scores['tone']={'anger':round(tone_score[0,0],4),'fear':round(tone_score[0,1],4),'joy':round(tone_score[0,2],4),'love':round(tone_score[0,3],4),'sadness':round(tone_score[0,4],4),'surprise':round(tone_score[0,5],4)}
        sentenceScores.append(scores)
    return sentenceScores

def getBadSentences(sentenceScores, sentences):
    result = []
    i = 0
    while i < len(sentenceScores):
        errors = ""
        if sentenceScores[i]["politeness"]["rude"] > 0.6:
            errors += "rude"
        elif sentenceScores[i]['word_count'] > 25:
            errors += "length"
        result.append([errors, sentences[i][-1], sentences[i].split()[0]])
        i += 1
    return result

def getPunctFreeString(list):
    str1=''
    for word in list:
        if word not in string.punctuation:
            str1+=word+' '
    str1=str1[:len(str1)-1]
    return str1

def getComplexWords(text):
    ncomplex=0
    ntotal = 0
    complexlist = []
    #words that the algorithm inorrectly does not classify as complex
    exceptions = ["amazing", "terrible", "horrible", "laughable", "countable", "probable", "constable", "capable", "audible", "visible", "breakable", "flexible", "plausible", "tangible", "feasible", "palpable", "flammable", "unstable", "winnable", "losable", "mashable", "flappable", "effable", "bendable", "idiot", "arable"]
    #words that the algorithm incorrectly classifies as complex
    exceptions2 = ["sometime", "enclosed", "unique", "aligned", "received", "business", "useful"]
    for word in text:
        syllables = 0
        for i in range(len(word)):
            if i == 0 and word[i] in "aeiouy" :
                syllables = syllables + 1
            elif word[i - 1] not in "aeiouy" :
                if i < len(word) - 1 and word[i] in "aeiouy" :
                    syllables = syllables + 1
                elif i == len(word) - 1 and word[i] in "aiouy" :
                    syllables = syllables + 1
        if len(word) > 0 and syllables == 0 :
            syllables = 1
        if (syllables>=3 or word in exceptions) and (word not in exceptions2):
            ncomplex=ncomplex+1
            complexlist.append(word)
        ntotal += syllables
    return ncomplex, ntotal, complexlist


def getOverallScore(scores):
    total_score = 0
    love_score = max(scores["tone"]["love"] - 0.167, 0);
    joy_score = max(scores["tone"]["joy"] - 0.167, 0);
    surprise_score = max(scores["tone"]["surprise"] - 0.167, 0);
    fear_score = max(scores["tone"]["fear"] - 0.167, 0);
    anger_score = max(scores["tone"]["anger"] - 0.167, 0);
    sadness_score = max(scores["tone"]["sadness"] - 0.167, 0);
    tone_score = love_score + joy_score + surprise_score + fear_score + anger_score + sadness_score;
    score_polite = scores["politeness"]["polite"]*100
    #give emails with strong love, joy, and surprise a higher overall score
    score_tone = ((love_score/tone_score) + (joy_score/tone_score) + (surprise_score/tone_score))*100
    score_complexity = (100 - ((max(scores["complexity"] - 50, 0))))
    score_positivity = ((scores["compound"] + 1)/2)*100

    total_score = score_polite*0.25 + score_tone*0.25 + score_complexity*0.25 + score_positivity*0.25
    return round(total_score)

if __name__ == '__main__':
    app.run(debug = True)
