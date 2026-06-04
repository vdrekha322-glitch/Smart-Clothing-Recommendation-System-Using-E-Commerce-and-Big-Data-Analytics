from flask import Flask, request, jsonify
from flask_cors import CORS
from model import predict_size
import csv
import io

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    size = predict_size(data)
    return jsonify({"recommended_size": size})

# 🔹 FILE BASED INPUT (CSV)
@app.route("/predict-file", methods=["POST"])
def predict_file():
    file = request.files["file"]
    stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
    reader = csv.DictReader(stream)

    results = []
    for row in reader:
        size = predict_size(row)
        row["recommended_size"] = size
        results.append(row)

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
