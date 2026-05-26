from typing import List
from flask import Flask, request, jsonify
import torch
import numpy as np
from pathlib import Path
from modelos import CNN1Conv

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints")

def carregar_modelo_e_norm(sensor):
    modelo = CNN1Conv(8).to(device)
    modelo.load_state_dict(torch.load(CHECKPOINT_DIR / sensor / f"{sensor}_FINAL.pth", map_location=device))
    modelo.eval()
    
    mean = np.load(CHECKPOINT_DIR / sensor / f"{sensor}_FINAL_mean.npy")
    std = np.load(CHECKPOINT_DIR / sensor / f"{sensor}_FINAL_std.npy")
    return modelo, mean, std

# Carregando os modelos e seus normalizadores
modelo_chest, mean_chest, std_chest = carregar_modelo_e_norm("CHEST")
modelo_left, mean_left, std_left = carregar_modelo_e_norm("LEFT")
modelo_right, mean_right, std_right = carregar_modelo_e_norm("RIGHT")

def preparar_tensor(janela: List[List[float]], mean: np.ndarray, std: np.ndarray):
    x = np.array(janela, dtype=np.float32)
    x = (x - mean) / std
    x = x.transpose(1, 0)
    x = torch.tensor(x).unsqueeze(0).to(device)
    return x

def prever_probs(modelo, janela, mean, std):
    x = preparar_tensor(janela, mean, std)
    with torch.no_grad():
        logits = modelo(x)
        probs = torch.softmax(logits, dim=1)
    return probs

def ensemble_predict(chest, left, right):
    probs_chest = prever_probs(modelo_chest, chest, mean_chest, std_chest)
    probs_left = prever_probs(modelo_left, left, mean_left, std_left)
    probs_right = prever_probs(modelo_right, right, mean_right, std_right)
    
    probs_final = (probs_chest + probs_left + probs_right) / 3.0
    classe_final = torch.argmax(probs_final, dim=1).item()
    
    return classe_final, probs_final.squeeze().cpu().tolist()

@app.route("/receber", methods=["POST"])
def receber():
    try:
        data = request.get_json()

        timestamp = data.get("timestamp", "0")
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