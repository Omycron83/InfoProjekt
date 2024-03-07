from flask import Flask, render_template, request, redirect, url_for
import Datenbankprojekt
import autofill
from DecisionTrees import DecisionTreeAutoRegr,DecisionTreeAutoClass
from k_means import KMeansClustering
from LinRegr import PolynomialRegrAuto
from LogRegr import LogisticAutoClass
from NeuralNetwork import NeuralNetworkAutoClass, NeuralNetworkAutoRegr
import outlier
from PCA import PCA
from RandomForests import RandomForestAutoClass, RandomForestAutoRegr
from XGBoost import XGBoostAutoClass, XGBoostAutoRegr

app = Flask(__name__)
manager = Datenbankprojekt("Databank-Manager")
manager.DatenbankErstellen("Modelle")

manager.DatenbankErstellen("Datensaetze")
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
    data, labels = ...
    #Tree model for classification
    Models = [RandomForestAutoClass(data, labels), XGBoostAutoClass(data, labels), DecisionTreeAutoClass(data, labels)]
    Models.sort(key= lambda x: x.cost(data, labels))
    Model = Models[0]
    save the model
    pass

@app.route('/supervised/structured/regression')
def structured_regression():
    data, labels = ...
    #Tree model for regression
    Models = [RandomForestAutoRegr(data, labels), XGBoostAutoRegr(data, labels), DecisionTreeAutoRegr(data, labels)]
    Models.sort(key= lambda x: x.cost(data, labels))
    Model = Models[0]
    save the model
    pass

@app.route('/supervised/unstructured/classification')
def unstructured_regression():
    data, labels = ...
    #Linear or NN regression model
    Models = [PolynomialRegrAuto(data, labels), NeuralNetworkAutoRegr(data, labels)]
    Models.sort(key= lambda x: x.cost(data, labels))
    Model = Models[0]
    save the model
    pass

@app.route('/supervised/unstructured/regression')
def unstructured_classification():
    data, labels = ...
    #Linear or NN classification model
    Models = [LogisticAutoClass(data, labels), NeuralNetworkAutoClass(data, labels)]
    Models.sort(key= lambda x: x.cost(data, labels))
    Model = Models[0]
    save the model
    pass

@app.route('/unsupervised/processing')
def unsupervised_processing():
    pass

@app.route('/unsupervised/outlier')
def unsupervised_outliers():
    data = ...
    Fraktilwert = ...
    #Filtering out outliers
    Filter = outlier.outlier_filtering(Fraktilwert)
    data = Filter.filter(data)
    save the data
    pass

@app.route('/unsupervised/cond')
def unsupervised_cond():
    amt_dim = ...
    data = ...
    #Save data pca
    PCA_Model = PCA(amt_dim)
    data_cond = PCA_Model.pca(data)
    save the data
    pass

@app.route('/unsupervised/cluster')
def unsupervised_cluster():
    amt_categories = ...
    data = ...
    #Save clusters of data
    Cluster_Model = KMeansClustering(amt_categories, data)
    classifications = Cluster_Model.get_labels(data)
    save the data
    pass

@app.route('/unsupervised/imputation')
def unsupervised_imputation():
    pass

@app.route('/unsupervised/imputation/mean')
def unsupervised_imputation_mean():
    data = ...
    #Save data filled with mean values
    data = autofill.mean_imputation(data)
    Save the data
    pass

@app.route('/unsupervised/imputation/kmeans')
def unsupervised_imputation_kmeans():
    data = ...
    #Save data filled with kmeans values
    data = autofill.k_nn_imputation(data)
    Save the data
    pass

@app.route('/existing')
def predict_reconfigure_download():
    OtpimModel = ... #Retrieve the model
    data, labels = ... #New stuff to reconfigure
    if OtpimModel.isinstance(RandomForestAutoClass) or OtpimModel.isinstance(XGBoostAutoClass) or OtpimModel.isinstance(DecisionTreeAutoClass):
        go to structured_classification
        pass
    elif OtpimModel.isinstance(RandomForestAutoRegr) or OtpimModel.isinstance(XGBoostAutoRegr) or OtpimModel.isinstance(DecisionTreeAutoRegr):
        go to structured_regression
        pass
    elif OptimModel.isinstance(PolynomialRegrAuto) or OptimModel.isinstance(NeuralNetworkAutoRegr):
        go to unstructured_regression
        pass
    elif OptimModel.isinstance(LogisticAutoClass) or OptimModel.isinstance(NeuralNetworkAutoClass):
        go to unstructured_classification
        pass

@app.route('/existing/pred')
def existing_predict():
    OptimModel = ...
    if OptimModel.isinstance(XGBoostAutoClass) or OptimModel.isinstance(RandomForestAutoClass) or OptimModel.isinstance(LogisticAutoClass) or OptimModel.isinstance(DecisionTreeAutoClass) or OptimModel.isinstance(NeuralNetworkAutoClass):
        prediction = ...
        prediction = prediction > 0.5 #If the dimension is just 0
    if OptimModel.isinstance(XGBoostAutoRegr) or OptimModel.isinstance(RandomForestAutoRegr) or OptimModel.isinstance(PolynomialRegrAuto) or OptimModel.isinstance(DecisionTreeAutoRegr) or OptimModel.isinstance(NeuralNetworkAutoRegr):
        prediction = ...
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