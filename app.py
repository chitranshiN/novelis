from flask import Flask, request
import joblib

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return {"message": "Hello World!"}


model = joblib.load(open("model_ada.pkl", "rb"))
columns = joblib.load(open("model_columns.pkl", "rb"))


@app.route("/api", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    try:
        filtered_data = []
        for column in columns:
            filtered_data.append(data[column])

    except KeyError:
        return {"error": f"Some data is missing from {columns}"}

    prediction = model.predict([filtered_data])
    output = prediction[0]
    return {"prediction": int(output)}
