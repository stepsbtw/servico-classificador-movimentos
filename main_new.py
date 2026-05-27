from typing import Dict, List
from flask import Flask, request, jsonify
import torch
import numpy as np
from pathlib import Path
from modelos import CNN1Conv

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints")
COMBINACOES = {("chest", "left", "right"): ("ALL", 24), ("chest", "left"): ("CHEST_LEFT", 16), ("chest", "right"): ("CHEST_RIGHT", 16), ("left", "right"): ("LEFT_RIGHT", 16), ("chest",): ("CHEST", 8), ("left",): ("LEFT", 8), ("right",): ("RIGHT", 8)}

def carregar_modelo_e_norm(nome_modelo: str, num_features: int):
    modelo = CNN1Conv(num_features).to(device)
    modelo.load_state_dict(torch.load(CHECKPOINT_DIR / nome_modelo / f"{nome_modelo}_FINAL.pth", map_location=device))
    modelo.eval()
    mean = np.load(CHECKPOINT_DIR / nome_modelo / f"{nome_modelo}_FINAL_mean.npy")
    std = np.load(CHECKPOINT_DIR / nome_modelo / f"{nome_modelo}_FINAL_std.npy")
    return modelo, mean, std

modelos_em_memoria = {}
for combo, (nome_modelo, num_features) in COMBINACOES.items():
    try:
        modelos_em_memoria[combo] = carregar_modelo_e_norm(nome_modelo, num_features)
    except Exception:
        pass

def construir_janela(dados_sensor: Dict) -> np.ndarray:
    acc = np.array(dados_sensor["linear_acceleration"], dtype=np.float32)
    gyro = np.array(dados_sensor["angular_speed"], dtype=np.float32)
    amag = np.sqrt(np.sum(np.square(acc), axis=1, keepdims=True))
    wmag = np.sqrt(np.sum(np.square(gyro), axis=1, keepdims=True))
    return np.concatenate((acc, amag, gyro, wmag), axis=1)

@app.route("/receber", methods=["POST"])
def receber():
    try:
        data = request.get_json()
        timestamp = data.get("timestamp", "0")
        sensores_validos = []
        janelas_individuais = []
        for sensor_name in ["chest", "left", "right"]:
            sensor_data = data.get(sensor_name)
            if sensor_data and len(sensor_data.get("linear_acceleration", [])) == 180 and len(sensor_data.get("angular_speed", [])) == 180:
                sensores_validos.append(sensor_name)
                janelas_individuais.append(construir_janela(sensor_data))
        if not sensores_validos:
            return jsonify({"erro": "Nenhum dado valido"}), 400
        chave_combo = tuple(sensores_validos)
        if chave_combo not in modelos_em_memoria:
            return jsonify({"erro": f"Modelo nao carregado para {chave_combo}"}), 500
        modelo_selecionado, mean, std = modelos_em_memoria[chave_combo]
        nome_modelo_acionado = COMBINACOES[chave_combo][0]
        janela_fusion = np.concatenate(janelas_individuais, axis=1)
        x = (janela_fusion - mean) / std
        x = x.transpose(1, 0)
        x = torch.tensor(x).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = modelo_selecionado(x)
            probs = torch.softmax(logits, dim=1)
        classe_final = torch.argmax(probs, dim=1).item()
        probs_list = probs.squeeze().cpu().tolist()
        return jsonify({"timestamp": timestamp, "classe_predita": classe_final, "probabilidades": probs_list, "sensores_utilizados": sensores_validos, "modelo_acionado": nome_modelo_acionado, "device": str(device)})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
