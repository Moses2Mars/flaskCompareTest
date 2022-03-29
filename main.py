from flask import Flask
from flask import request
import docx2txt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
app = Flask(__name__)


@app.route('/')
def hello():
    return '<h1>Hello, World!</h1>'

@app.route('/cv-form', methods=['POST'])
async def form_example():
    if request.method == 'POST':
        email = request.form.get('email')
        job_uuid = request.form.get('job_uuid')
        job_description = request.form.get('job_description')
        CV_TEST = request.files['file']
        resume = docx2txt.process(CV_TEST)
        text_resume = str(resume)
        text_description = str(job_description)
        text_list = [text_resume, text_description] 
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(text_list)
        matchPercentage = cosine_similarity(count_matrix)[0][1] * 100
        matchPercentage = round(matchPercentage, 2) # round to two decimal
        print('Your resume matches about '+ str(matchPercentage)+ " percent of the job description.")
        return '''
                    <h1>The email is: {}</h1>
                    <h1>The cv is: {}</h1>
                        '''.format(email, CV_TEST)
