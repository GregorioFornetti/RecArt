from flask import Flask
from flask import render_template
from flask import request
from flask import send_from_directory
from werkzeug.utils import secure_filename
import os

import sys
sys.path.append("..")
from models.utils.dataset_preprocess import preprocess_dataset
from models.rec_models.RecSimpleNeuralNetwork import RecSimpleNeuralNetwork
from models.rec_models.RecAlexNet import RecAlexNet
from models.rec_models.RecKNN import RecKNN
from models.rec_models.RecResNet50 import RecResNet50

preprocess_dataset()

recommenders = {
    'simple-net': RecSimpleNeuralNetwork(),
    'alexnet': RecAlexNet(),
    #'knn': RecKNN(),
    'resnet': RecResNet50()
}

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route('/users_images/<path:filename>', methods=["GET"])
def users_images(filename):
    return send_from_directory("users_images", filename, as_attachment=True)

@app.route('/dataset_images/<path:dirname>/<path:filename>', methods=["GET"])
def dataset_images(dirname, filename):
    return send_from_directory(f"../../dataset/images/images/{dirname}", filename, as_attachment=True)

@app.route("/recomendar", methods=["POST"])
def classificar():

    image = request.files['upload_image']
    image_path = f"users_images/{secure_filename(image.filename)}"
    image.save(image_path)

    recs = recommenders[request.form['select-net']].predict(image_path)

    return render_template("recomendacoes.html", 
                           image_name=secure_filename(image.filename), 
                           selected_genres=recs['predictions']['selected'],
                           not_selected_genres=recs['predictions']['not selected'],
                           similar_images=recs['similar images'].iloc[:20],
                           path_join=os.sep)

