import joblib as jb
import pandas as pd
from flask import Flask, render_template, request
from sklearn.pipeline import make_pipeline

best_tree_load = jb.load("./predict_car.joblib")
preprocessador_load = jb.load("./preprocessador.joblib")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def novos_dados():
    req = request.form
    

    registro = pd.DataFrame(
        data=[[req["compra"], req["manutencao"], req["portas"], req["pessoas"], req["porta-malas"], req["seguranca"]]],
        columns=["compra", "manutencao", "portas", "pessoas", "porta-malas", "seguranca"]
    )

    pipeline = make_pipeline(preprocessador_load, best_tree_load)
    pipeline.fit(registro)
    registro_encode = preprocessador_load.transform(registro)
    registro["avaliacao"] = best_tree_load.predict(registro_encode)

    # Extrair a avaliação para passar para o template
    if registro["avaliacao"].values[0]=="unacc":
        avaliacao = "unacceptable"
    elif registro["avaliacao"].values[0]=="acc":
        avaliacao = "acceptable"
    elif registro["avaliacao"].values[0]=="good":
        avaliacao = "good"
    elif registro["avaliacao"].values[0]=="vgood":
        avaliacao = "very good"
    

    return render_template('index.html', retorno=req, avaliacao=avaliacao)

app.run(debug=True, port=5000)
