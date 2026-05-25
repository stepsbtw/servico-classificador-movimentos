from typing import List
from flask import Flask, request, jsonify
import torch
import numpy as np

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modelo_chest = torch.load("chest.pt", map_location=device).to(device)
modelo_left = torch.load("left.pt", map_location=device).to(device)
modelo_right = torch.load("right.pt", map_location=device).to(device)

modelo_chest.eval()
modelo_left.eval()
modelo_right.eval()

def preparar_tensor(janela: List[float]):
    x = np.array(janela, dtype=np.float32)
    x = torch.tensor(x).unsqueeze(0).to(device)
    return x

def prever_probs(modelo, janela):
    x = preparar_tensor(janela)
    with torch.no_grad():
        logits = modelo(x)
        probs = torch.softmax(logits, dim=1)
    return probs

def ensemble_predict(chest, left, right):
    probs_chest = prever_probs(modelo_chest, chest)
    probs_left = prever_probs(modelo_left, left)
    probs_right = prever_probs(modelo_right, right)
    probs_final = (probs_chest + probs_left + probs_right) / 3.0
    classe_final = torch.argmax(probs_final, dim=1).item()
    return classe_final, probs_final.squeeze().cpu().tolist()

@app.route("/receber", methods=["POST"])
def receber():
    try:
        data = request.get_json()

        timestamp = data["timestamp"]
        chest = data["chest"]
        left = data["left"]
        right = data["right"]

        if len(chest) != 180 or len(left) != 180 or len(right) != 180:
            return jsonify({"erro": "cada entrada deve ter 180 amostras"}), 400

        classe, probs = ensemble_predict(chest, left, right)

        return jsonify({
            "timestamp": timestamp,
            "classe_predita": classe,
            "probabilidades": probs,
            "device": str(device)
        })

    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)