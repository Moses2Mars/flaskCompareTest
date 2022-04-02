from flask import Flask
from flask import request
import docx2txt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pdfplumber
import os

app = Flask(__name__)


@app.route('/')
def hello():
    return '<h1>Hello, World!</h1>'

@app.route('/cv-form', methods=['POST'])
async def form_example():
    if request.method == 'POST':
        #job_description is already a string, to be compared with the CV
        job_description = request.form.get('job_description')
        
        #CV file from the request
        user_cv = request.files['file']
        extension = user_cv.filename.split('.')
        #if type is PDF
        if extension[-1] == 'pdf':
            #change from docx format to text (string) format
            resume = readFromPDF(user_cv)
        else: 
            #if type is doc/docx
            resume = docx2txt.process(user_cv)  

        #making sure job description is in string format
        text_resume = str(resume)
        text_resume = cleanString(text_resume)
        
        #making sure job description is in string format
        text_description = str(job_description)
        text_description = cleanString(text_description)

        #making a list of both the resume and job description
        text_list = [text_resume, text_description]

        # Convert a collection of text documents to a matrix of token counts 
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(text_list)

        # Cosine similarity is a metric used to measure how similar the documents are irrespective of their size. 
        # Mathematically, it measures the cosine of the angle between two vectors projected in a multi-dimensional space. 
        # The cosine similarity is advantageous because even if the two similar documents are far apart by the Euclidean distance 
        # (due to the size of the document), chances are they may still be oriented closer together. 
        # The smaller the angle, the higher the cosine similarity.
        matchPercentage = cosine_similarity(count_matrix)[0][1] * 100
        matchPercentage = round(matchPercentage, 2) # round to two decimal

        print(str(matchPercentage))
        return str(matchPercentage)


def readFromPDF(file): 
    text = ''
    with pdfplumber.open(file) as pdf:
        for pdf_page in pdf.pages:
            text += pdf_page.extract_text()
    return text

def cleanString(stringer): 
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(stringer)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    import string
    table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    words = ' '.join(words)
    return str(words)
