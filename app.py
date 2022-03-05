from flask import Flask, render_template, request, url_for
import pickle

filename = 'rfc_model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv-transform.pkl','rb'))
app = Flask(__name__)

@app.route("/", endpoint='func1')
def main_page():
    return render_template('home.html')

@app.route("/news_page", endpoint='func3')
def news():
    return render_template('news.html')

@app.route("/contact_page", endpoint='func4')
def contact():
    return render_template('contact.html')
    
@app.route("/predict", methods = ['POST', 'GET'], endpoint='func2')
def predict():
    if request.method == 'POST':
        n = request.form['news']
        data = [n]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        return render_template('predicted.html', prediction = my_prediction)

if __name__ ==  "__main__":
    app.run(debug=True)