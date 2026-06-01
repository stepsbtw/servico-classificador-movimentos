from typing import Dict
from flask import Flask, request, jsonify
import torch
import numpy as np
import json
from pathlib import Path
from modelos import CNN1Conv

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints")

try:
    with open("mapping.json", "r") as f:
        mapping_data = json.load(f)
        
    MAP_POSTURE = {int(k): v[0] for k, v in mapping_data.get("y_classify_posture", {}).items()}
    MAP_MOVEMENT = {int(k): v[0] for k, v in mapping_data.get("y_classify_movement", {}).items()}
except FileNotFoundError:
    print("Aviso: mapping.json não encontrado. A usar mapeamento de fallback.")
    MAP_POSTURE, MAP_MOVEMENT = {}, {}

COMBINACOES = {
    ("chest", "left", "right"): ("CHEST_LEFT_RIGHT", 24), 
    ("chest", "left"): ("CHEST_LEFT", 16), 
    ("chest", "right"): ("CHEST_RIGHT", 16), 
    ("left", "right"): ("LEFT_RIGHT", 16), 
    ("chest",): ("CHEST", 8), 
    ("left",): ("LEFT", 8), 
    ("right",): ("RIGHT", 8)
}

def carregar_modelo_e_norm(tarefa: str, nome_modelo: str, num_features: int):
    if tarefa == "y_detect_fall": num_classes = 2
    elif tarefa == "y_classify_posture": num_classes = len(MAP_POSTURE) if MAP_POSTURE else 4
    elif tarefa == "y_classify_movement": num_classes = len(MAP_MOVEMENT) if MAP_MOVEMENT else 5
    else: num_classes = 2

    modelo = CNN1Conv(num_features, num_classes=num_classes).to(device)
    caminho_base = CHECKPOINT_DIR / tarefa / nome_modelo
    modelo.load_state_dict(torch.load(caminho_base / f"{nome_modelo}_FINAL.pth", map_location=device))
    modelo.eval()
    mean = np.load(caminho_base / f"{nome_modelo}_FINAL_mean.npy")
    std = np.load(caminho_base / f"{nome_modelo}_FINAL_std.npy")
    return modelo, mean, std

modelos_queda, modelos_postura, modelos_movimento = {}, {}, {}

for combo, (nome_modelo, n_feat) in COMBINACOES.items():
    try: modelos_queda[combo] = carregar_modelo_e_norm("y_detect_fall", nome_modelo, n_feat)
    except Exception: pass
    try: modelos_postura[combo] = carregar_modelo_e_norm("y_classify_posture", nome_modelo, n_feat)
    except Exception: pass
    try: modelos_movimento[combo] = carregar_modelo_e_norm("y_classify_movement", nome_modelo, n_feat)
    except Exception: pass

def construir_janela(dados_sensor: Dict) -> np.ndarray:
    acc = np.array(dados_sensor["linear_acceleration"], dtype=np.float32)
    gyro = np.array(dados_sensor["angular_speed"], dtype=np.float32)
    amag = np.sqrt(np.sum(np.square(acc), axis=1, keepdims=True))
    wmag = np.sqrt(np.sum(np.square(gyro), axis=1, keepdims=True))
    return np.concatenate((acc, amag, gyro, wmag), axis=1)

def prever_early_fusion(modelos_dict, chave_combo, janelas_individuais):
    if chave_combo not in modelos_dict:
        raise ValueError(f"Modelo para a combinação {chave_combo} não encontrado.")
        
    modelo, mean, std = modelos_dict[chave_combo]
    janela_fusion = np.concatenate(janelas_individuais, axis=1)
    
    x = (janela_fusion - mean) / std
    x = x.transpose(1, 0)
    x = torch.tensor(x).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = modelo(x)
        probs = torch.softmax(logits, dim=1)

    classe_final = torch.argmax(probs, dim=1).item()
    return classe_final, probs.squeeze().cpu().tolist()

@app.route("/receber", methods=["POST"])
def receber():
    try:
        data = request.get_json()
        timestamp = data.get("timestamp", "0")
        
        sensores_validos = []
        janelas_individuais = []

        for sensor_name in ["chest", "left", "right"]:
            sensor_data = data.get(sensor_name)
            if sensor_data and len(sensor_data.get("linear_acceleration", [])) == 180:
                sensores_validos.append(sensor_name)
                janelas_individuais.append(construir_janela(sensor_data))

        if not sensores_validos:
            return jsonify({"erro": "Nenhum dado válido"}), 400

        chave_combo = tuple(sensores_validos)

        classe_queda, probs_queda = prever_early_fusion(modelos_queda, chave_combo, janelas_individuais)
        classe_postura, probs_postura = prever_early_fusion(modelos_postura, chave_combo, janelas_individuais)
        classe_movimento, probs_movimento = prever_early_fusion(modelos_movimento, chave_combo, janelas_individuais)

        print(f"\nModel Fusion: {COMBINACOES[chave_combo][0]}")
        
        lbl_fall = "FALL DETECTED" if classe_queda == 0 else "NO FALL"
        pct_fall = max(probs_queda) * 100
        print(f"Detect Fall: {lbl_fall} ({pct_fall:.1f}%)")
        
        lbl_posture = MAP_POSTURE.get(classe_postura, "IGNORE")
        pct_posture = max(probs_postura) * 100
        print(f"Classify Posture: {lbl_posture} ({pct_posture:.1f}%)")
        
        lbl_movement = MAP_MOVEMENT.get(classe_movimento, "IGNORE")
        pct_movement = max(probs_movimento) * 100
        print(f"Classify Movement: {lbl_movement} ({pct_movement:.1f}%)")
        print("-" * 40)

        resultado = {
            "timestamp": timestamp,
            "sensores_utilizados": sensores_validos,
            "modelo_early_fusion": COMBINACOES[chave_combo][0],
            "detect_fall": {"classe": classe_queda, "probabilidades": probs_queda},
            "classify_posture": {"classe": classe_postura, "probabilidades": probs_postura},
            "classify_movement": {"classe": classe_movimento, "probabilidades": probs_movimento}
        }
        return jsonify(resultado)

    except ValueError as ve:
        return jsonify({"erro": str(ve)}), 400
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)