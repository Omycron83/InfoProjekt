from flask import Flask, render_template, request, redirect, url_for
import Datenbankprojekt
import autofill
import DecisionTrees
import k_means
import LinRegr
import LogRegr
import NeuralNetwork
import outlier
import PCA
import RandomForests
import XGBoost

app = Flask(__name__)

#Decision to either decide on new or existing
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/supervised-unsupervised')
def super_unsuper():
    pass

@app.route('/supervised/prep-data')
def prep_data():
    pass

@app.route('/supervised/structured-unstructured')
def structured():
    pass

@app.route('/supervised/structured/classification')
def structured_classification():
    pass

@app.route('/supervised/structured/regression')
def structured_regression():
    pass

@app.route('/supervised/unstructured/classification')
def unstructured_regression():
    pass

@app.route('/supervised/unstructured/regression')
def unstructured_classification():
    pass

@app.route('/unsupervised/processing')
def unsupervised_processing():
    pass

@app.route('/unsupervised/cond')
def unsupervised_cond():
    pass

@app.route('/unsupervised/cluster')
def unsupervised_cluster():
    pass

@app.route('/unsupervised/imputation')
def unsupervised_imputation():
    pass

@app.route('/unsupervised/imputation/mean')
def unsupervised_imputation_mean():
    pass

@app.route('/unsupervised/imputation/kmeans')
def unsupervised_imputation_kmeans():
    pass

@app.route('/existing')
def predict_reconfigure_download():
    pass

@app.route('/existing/pred')
def existing_predict():
    pass

@app.route('/existing/conf')
def existing_configure():
    pass

@app.route('/existing/download')
def existing_download():
    pass

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        button_pressed = request.form.get('button_pressed', False)
        return render_template('result.html', button_pressed=button_pressed)

if __name__ == "__main__":
    app.run()