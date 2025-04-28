# Importing essential libraries
from py_compile import main
from flask import Flask, render_template, request
import pickle

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = 'voting_clf.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('countvector.pkl','rb'))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        message = request.form['message']  # Fetch the input from the form
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        return render_template('result.html', prediction=my_prediction)
    return render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True)