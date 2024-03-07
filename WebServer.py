from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
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
import os

#Setting up the application
app = Flask(__name__)
#
manager = Datenbankprojekt.Datenbanken("Databank-Manager")
manager.DatenbankErstellen("Datenbank")
manager.TabellenErstellen("HauptTabelle",[["ProjectName",""],["DataArray","array"],["LabelArray","array"]])
manager.TabellenErstellen("ModelStructured_Classification",[["ProjectName",""],["model","Damianstuff"]])
manager.TabellenErstellen("ModelStructured_Regression",[["ProjectName",""],["model","Damianstuff"]])
manager.TabellenErstellen("ModelUnstructured_Regression",[["ProjectName",""],["model","Damianstuff"]])
manager.TabellenErstellen("ModelUnstructured_Classification",[["ProjectName",""],["model","Damianstuff"]])


UPLOAD_FOLDER = 'UPLOAD_FOLDER'
ALLOWED_EXTENSIONS = {'csv'}

from flask import Flask, render_template, request, redirect

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello_world():  # put application's code here
    if request.method == 'POST':
        username = request.form['username']
        print(username)
        return redirect('/home')
    return render_template('index.html')


@app.route('/home')
def home():
    return render_template('home.html', datasets=["airplane", "automobile", "bird", "cat"])


@app.route('/download/<name>')
def download(name):
    print(name)
    return redirect('/home')


@app.route('/create', methods=['GET', 'POST'])
def create():
    if request.method == 'POST':
        name = request.form['name']
        print(name)
        if 'preparation' in request.form and request.form['preparation'] == 'on':
            return redirect('/preparation')
        return redirect('/unsupervised')
    return render_template('create.html')


@app.route('/preparation', methods=['GET', 'POST'])
def prepare_data():
    if request.method == 'POST':
        check = lambda key: key in request.form and request.form[key] == 'on'
        selected_typ = ''
        for typ in ['condensation', 'clustering', 'imputation-mean', 'imputation-knn']:
            if check(typ):
                selected_typ = typ
                break
        print(selected_typ)
        print(request.files)
        if 'skip' in request.form and request.form['skip'] == 'on':
            return redirect('/home')
        return redirect('/unsupervised')
    return render_template('preparation.html')


@app.route('/unsupervised', methods=['GET', 'POST'])
def unsupervised():
    if request.method == 'POST':
        structured = 'structured' in request.form and request.form['structured'] == 'on'
        print(structured)
        check = lambda key: key in request.form and request.form[key] == 'on'
        selected_typ = ''
        for typ in ['class', 'reg']:
            if check(typ):
                selected_typ = typ
                break
        print(selected_typ)
        print(request.files['file'].filename)
        return redirect('/home')
    return render_template('unsupervised.html')


@app.route('/predict/<name>', methods=['GET', 'POST'])
def predict(name):
    if request.method == 'POST':
        print(request.files)
        # DO DOWNLOAD
        return redirect('/home')
    return render_template('predict.html')


if __name__ == '__main__':
    app.run()

# @app.route('/Upload-Data')
# def upload_file():
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         # If the user does not select a file, the browser submits an
#         # empty file without a filename.
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(os.path.join(app.root_path, app.config['UPLOAD_FOLDER']), filename))
#             return redirect(url_for('download_file', name=filename))
#     return '''
#     <!doctype html>
#     <title>Upload new File</title>
#     <h1>Upload new File</h1>
#     <form method=post enctype=multipart/form-data>
#       <input type=file name=file>
#       <input type=submit value=Upload>
#     </form>
#     '''

# @app.route('/supervised-unsupervised')
# def super_unsuper():
#     return render_template()


# @app.route('/supervised/prep-data')
# def prep_data():
#     pass

# @app.route('/supervised/structured-unstructured')
# def structured():
#     pass

# @app.route('/supervised/structured/classification')
# def structured_classification():
#     if request.method == 'POST':
#         do something damian
#     data, labels = manager.VonTabelleGebe("HauptTabelle",["DataArray","LabelArray"],"DataArray",["ProjectName"],["=="],["Hello World"])[0]
#     #Tree model for classification
#     Models = [RandomForestAutoClass(data, labels), XGBoostAutoClass(data, labels), DecisionTreeAutoClass(data, labels)]
#     Models.sort(key= lambda x: x.cost(data, labels))
#     Model = Models[0]
#     manager.TabellenInsert("ModelStructed_Classification",["Hello World",Model])
#     return render_template()

# @app.route('/supervised/structured/regression')
# def structured_regression():
#     data, labels = manager.VonTabelleGebe("HauptTabelle",["DataArray","LabelArray"],"DataArray",["ProjectName"],["=="],["Hello World"])[0]
#     #Tree model for regression
#     Models = [RandomForestAutoRegr(data, labels), XGBoostAutoRegr(data, labels), DecisionTreeAutoRegr(data, labels)]
#     Models.sort(key= lambda x: x.cost(data, labels))
#     Model = Models[0]
#     manager.TabellenInsert("ModelStructed_Regression",["Hello World",Model])
#     pass

# @app.route('/supervised/unstructured/classification')
# def unstructured_regression():
#     data, labels = manager.VonTabelleGebe("HauptTabelle",["DataArray","LabelArray"],"DataArray",["ProjectName"],["=="],["Hello World"])[0]
#     #Linear or NN regression model
#     Models = [PolynomialRegrAuto(data, labels), NeuralNetworkAutoRegr(data, labels)]
#     Models.sort(key= lambda x: x.cost(data, labels))
#     Model = Models[0]
#     manager.TabellenInsert("ModelUnstructed_Regression",["Hello World",Model])
#     pass

# @app.route('/supervised/unstructured/regression')
# def unstructured_classification():
#     data, labels = manager.VonTabelleGebe("HauptTabelle",["DataArray","LabelArray"],"DataArray",["ProjectName"],["=="],["Hello World"])[0]
#     #Linear or NN classification model
#     Models = [LogisticAutoClass(data, labels), NeuralNetworkAutoClass(data, labels)]
#     Models.sort(key= lambda x: x.cost(data, labels))
#     Model = Models[0]
#     manager.TabellenInsert("ModelUnstructed_Classification",["Hello World",Model])
#     pass

# @app.route('/unsupervised/processing')
# def unsupervised_processing():
#     pass

# @app.route('/unsupervised/outlier')
# def unsupervised_outliers():
#     data = manager.VonTabelleGebe("HauptTabelle",["DataArray"],"DataArray",["ProjectName"],["=="],["Hello World"])[0]
#     Fraktilwert = ...
#     #Filtering out outliers
#     Filter = outlier.outlier_filtering(Fraktilwert)
#     data_1 = Filter.filter(data)
#     manager.TabelleUpdaten("HauptTabelle",["DataArray"],[data_1],["ProjectName"],["=="],["Hello World"])
#     pass

# @app.route('/unsupervised/cond')
# def unsupervised_cond():
#     amt_dim = ...
#     data = manager.VonTabelleGebe("HauptTabelle",["DataArray"],"DataArray",["ProjectName"],["=="],["Hello World"])[0]
#     #Save data pca
#     PCA_Model = PCA(amt_dim)
#     data_cond = PCA_Model.pca(data)
#     manager.TabelleUpdaten("HauptTabelle",["DataArray"],[data_cond],["ProjectName"],["=="],["Hello World"])
#     pass

# @app.route('/unsupervised/cluster')
# def unsupervised_cluster():
#     amt_categories = ...
#     data = manager.VonTabelleGebe("HauptTabelle",["DataArray"],"DataArray",["ProjectName"],["=="],["Hello World"])[0]
#     #Save clusters of data
#     Cluster_Model = KMeansClustering(amt_categories, data)
#     classifications = Cluster_Model.get_labels(data)
#     manager.TabelleUpdaten("HauptTabelle",["DataArray"],[classifications],["ProjectName"],["=="],["Hello World"])
#     pass

# @app.route('/unsupervised/imputation')
# def unsupervised_imputation():
#     pass

# @app.route('/unsupervised/imputation/mean')
# def unsupervised_imputation_mean():
#     data = manager.VonTabelleGebe("HauptTabelle",["DataArray"],"DataArray",["ProjectName"],["=="],["Hello World"])[0]
#     #Save data filled with mean values
#     data_2 = autofill.mean_imputation(data)
#     manager.TabelleUpdaten("HauptTabelle",["DataArray"],[data_2],["ProjectName"],["=="],["Hello World"])
#     pass

# @app.route('/unsupervised/imputation/kmeans')
# def unsupervised_imputation_kmeans():
#     data = manager.VonTabelleGebe("HauptTabelle",["DataArray"],"DataArray",["ProjectName"],["=="],["Hello World"])[0]
#     #Save data filled with kmeans values
#     data_3 = autofill.k_nn_imputation(data)
#     manager.TabelleUpdaten("HauptTabelle",["DataArray"],[data_3],["ProjectName"],["=="],["Hello World"])
#     pass

# @app.route('/existing')
# def predict_reconfigure_download():
#     if manager.VonTabelleGebe("ModelStructured_Classification",["model"],"model",["ProjectName"],["=="],["Hello World"]) != []:
#         ModelStructured_Classification = VonTabelleGebe("ModelStructured_Classification",["model"],"model",["ProjectName"],["=="],["Hello World"])[0]
#     else:
#         ModelStructured_Classification = None
#     if manager.VonTabelleGebe("ModelStructured_Regression",["model"],"model",["ProjectName"],["=="],["Hello World"]) != []:
#         ModelStructured_Regression = VonTabelleGebe("ModelStructured_Regression",["model"],"model",["ProjectName"],["=="],["Hello World"])[0]
#     else:
#         ModelStructured_Regression = None
#     if manager.VonTabelleGebe("ModelUnstructured_Regression",["model"],"model",["ProjectName"],["=="],["Hello World"]) != []:
#         ModelUnstructured_Regression = VonTabelleGebe("ModelUnstructured_Regression",["model"],"model",["ProjectName"],["=="],["Hello World"])[0]
#     else:
#         ModelUnstructured_Regression = None
#     if manager.VonTabelleGebe("ModelUnstructured_Classification",["model"],"model",["ProjectName"],["=="],["Hello World"]) != []:
#         ModelUnstructured_Classification = VonTabelleGebe("ModelUnstructured_Classification",["model"],"model",["ProjectName"],["=="],["Hello World"])[0]
#     else:
#         ModelUnstructured_Classification = None
#     OtpimModel = ... #Retrieve the 
#     data, labels = manager.VonTabelleGebe("HauptTabelle",["DataArray","LabelArray"],"DataArray",["ProjectName"],["=="],["Hello World"])[0] #New stuff to reconfigure

#     if OtpimModel.isinstance(RandomForestAutoClass) or OtpimModel.isinstance(XGBoostAutoClass) or OtpimModel.isinstance(DecisionTreeAutoClass):
#         go to structured_classification
#         pass
#     elif OtpimModel.isinstance(RandomForestAutoRegr) or OtpimModel.isinstance(XGBoostAutoRegr) or OtpimModel.isinstance(DecisionTreeAutoRegr):
#         go to structured_regression
#         pass
#     elif OptimModel.isinstance(PolynomialRegrAuto) or OptimModel.isinstance(NeuralNetworkAutoRegr):
#         go to unstructured_regression
#         pass
#     elif OptimModel.isinstance(LogisticAutoClass) or OptimModel.isinstance(NeuralNetworkAutoClass):
#         go to unstructured_classification
#         pass

# @app.route('/existing/pred')
# def existing_predict():
#     if manager.VonTabelleGebe("ModelStructured_Classification",["model"],"model",["ProjectName"],["=="],["Hello World"]) != []:
#         ModelStructured_Classification = VonTabelleGebe("ModelStructured_Classification",["model"],"model",["ProjectName"],["=="],["Hello World"])[0]
#     else:
#         ModelStructured_Classification = None
#     if manager.VonTabelleGebe("ModelStructured_Regression",["model"],"model",["ProjectName"],["=="],["Hello World"]) != []:
#         ModelStructured_Regression = VonTabelleGebe("ModelStructured_Regression",["model"],"model",["ProjectName"],["=="],["Hello World"])[0]
#     else:
#         ModelStructured_Regression = None
#     if manager.VonTabelleGebe("ModelUnstructured_Regression",["model"],"model",["ProjectName"],["=="],["Hello World"]) != []:
#         ModelUnstructured_Regression = VonTabelleGebe("ModelUnstructured_Regression",["model"],"model",["ProjectName"],["=="],["Hello World"])[0]
#     else:
#         ModelUnstructured_Regression = None
#     if manager.VonTabelleGebe("ModelUnstructured_Classification",["model"],"model",["ProjectName"],["=="],["Hello World"]) != []:
#         ModelUnstructured_Classification = VonTabelleGebe("ModelUnstructured_Classification",["model"],"model",["ProjectName"],["=="],["Hello World"])[0]
#     else:
#         ModelUnstructured_Classification = None
#     OptimModel = ...
#     if OptimModel.isinstance(XGBoostAutoClass) or OptimModel.isinstance(RandomForestAutoClass) or OptimModel.isinstance(LogisticAutoClass) or OptimModel.isinstance(DecisionTreeAutoClass) or OptimModel.isinstance(NeuralNetworkAutoClass):
#         prediction = ...
#         prediction = prediction > 0.5 #If the dimension is just 0
#     if OptimModel.isinstance(XGBoostAutoRegr) or OptimModel.isinstance(RandomForestAutoRegr) or OptimModel.isinstance(PolynomialRegrAuto) or OptimModel.isinstance(DecisionTreeAutoRegr) or OptimModel.isinstance(NeuralNetworkAutoRegr):
#         prediction = ...
#     pass

# @app.route('/existing/conf')
# def existing_configure():
#     pass


# @app.route('/result', methods=['POST'])
# def result():
#     if request.method == 'POST':
#         button_pressed = request.form.get('button_pressed', False)
#         return render_template('result.html', button_pressed=button_pressed)

if __name__ == "__main__":
    app.run()