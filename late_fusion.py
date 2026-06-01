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

def carregar_modelo_e_norm(tarefa: str, sensor: str):
    if tarefa == "y_detect_fall": num_classes = 2
    elif tarefa == "y_classify_posture": num_classes = len(MAP_POSTURE) if MAP_POSTURE else 4
    elif tarefa == "y_classify_movement": num_classes = len(MAP_MOVEMENT) if MAP_MOVEMENT else 5
    else: num_classes = 2 

    modelo = CNN1Conv(8, num_classes=num_classes).to(device)
    caminho_base = CHECKPOINT_DIR / tarefa / sensor
    
    modelo.load_state_dict(torch.load(caminho_base / f"{sensor}_FINAL.pth", map_location=device))
    modelo.eval()
    
    mean = np.load(caminho_base / f"{sensor}_FINAL_mean.npy")
    std = np.load(caminho_base / f"{sensor}_FINAL_std.npy")
    
    return modelo, mean, std

modelos_queda, modelos_postura, modelos_movimento = {}, {}, {}

for s in ["CHEST", "LEFT", "RIGHT"]:
    try: modelos_queda[s.lower()] = carregar_modelo_e_norm("y_detect_fall", s)
    except Exception: pass
    try: modelos_postura[s.lower()] = carregar_modelo_e_norm("y_classify_posture", s)
    except Exception: pass
    try: modelos_movimento[s.lower()] = carregar_modelo_e_norm("y_classify_movement", s)
    except Exception: pass

def construir_janela(dados_sensor: Dict) -> np.ndarray:
    acc = np.array(dados_sensor["linear_acceleration"], dtype=np.float32)
    gyro = np.array(dados_sensor["angular_speed"], dtype=np.float32)
    amag = np.sqrt(np.sum(np.square(acc), axis=1, keepdims=True))
    wmag = np.sqrt(np.sum(np.square(gyro), axis=1, keepdims=True))
    return np.concatenate((acc, amag, gyro, wmag), axis=1)

def preparar_tensor(janela: np.ndarray, mean: np.ndarray, std: np.ndarray):
    x = (janela - mean) / std
    x = x.transpose(1, 0)
    x = torch.tensor(x).unsqueeze(0).to(device)
    return x

def prever_probs(modelo, janela, mean, std):
    x = preparar_tensor(janela, mean, std)
    with torch.no_grad():
        logits = modelo(x)
        probs = torch.softmax(logits, dim=1)
    return probs

def calcular_ensemble_late_fusion(dados_requisicao, modelos_dict, sensores_validos):
    probs_acumuladas = None
    sensores_usados = []
    dict_probs_sensores = {}
    
    for sensor_name in sensores_validos:
        if sensor_name not in modelos_dict: continue 
            
        sensores_usados.append(sensor_name)
        modelo, mean, std = modelos_dict[sensor_name]
        janela = construir_janela(dados_requisicao[sensor_name])
        probs = prever_probs(modelo, janela, mean, std)
        
        probs_list = probs.squeeze().cpu().tolist()
        dict_probs_sensores[sensor_name] = probs_list
        
        if probs_acumuladas is None: probs_acumuladas = probs
        else: probs_acumuladas += probs

    if not sensores_usados:
        raise ValueError("Nenhum modelo disponível para os sensores nesta tarefa.")
        
    probs_final = probs_acumuladas / len(sensores_usados)
    probs_final_list = probs_final.squeeze().cpu().tolist()
    classe_final = torch.argmax(probs_final, dim=1).item()
    
    return classe_final, probs_final_list, dict_probs_sensores

def fmt_list(lst):
    return "[" + ", ".join([f"{x:.4f}" for x in lst]) + "]"

@app.route("/receber", methods=["POST"])
def receber():
    try:
        data = request.get_json()
        timestamp = data.get("timestamp", "0")
        
        sensores_validos = [
            s for s in ["chest", "left", "right"]
            if data.get(s) and len(data.get(s).get("linear_acceleration", [])) == 180
        ]
        
        if not sensores_validos:
            return jsonify({"erro": "Nenhum sensor válido com 180 amostras"}), 400

        cl_fall, p_fall_final, d_fall_sens = calcular_ensemble_late_fusion(data, modelos_queda, sensores_validos)
        cl_post, p_post_final, d_post_sens = calcular_ensemble_late_fusion(data, modelos_postura, sensores_validos)
        cl_move, p_move_final, d_move_sens = calcular_ensemble_late_fusion(data, modelos_movimento, sensores_validos)
        
        print("\nDetect Fall:")
        for s in ["chest", "left", "right"]:
            if s in d_fall_sens: print(f"  {s.capitalize()}: {fmt_list(d_fall_sens[s])}")
        lbl_fall = "FALL DETECTED" if cl_fall == 0 else "NO FALL"
        print(f"  Final: {lbl_fall} {fmt_list(p_fall_final)}")
        
        print("Classify Posture:")
        for s in ["chest", "left", "right"]:
            if s in d_post_sens: print(f"  {s.capitalize()}: {fmt_list(d_post_sens[s])}")
        lbl_posture = MAP_POSTURE.get(cl_post, "IGNORE")
        print(f"  Final: {lbl_posture} {fmt_list(p_post_final)}")
        
        print("Classify Movement:")
        for s in ["chest", "left", "right"]:
            if s in d_move_sens: print(f"  {s.capitalize()}: {fmt_list(d_move_sens[s])}")
        lbl_movement = MAP_MOVEMENT.get(cl_move, "IGNORE")
        print(f"  Final: {lbl_movement} {fmt_list(p_move_final)}")
        print("-" * 40)

        resultado = {
            "timestamp": timestamp,
            "sensores_utilizados": sensores_validos,
            "detect_fall": {"classe": cl_fall, "probabilidades": p_fall_final, "sensores": d_fall_sens},
            "classify_posture": {"classe": cl_post, "probabilidades": p_post_final, "sensores": d_post_sens},
            "classify_movement": {"classe": cl_move, "probabilidades": p_move_final, "sensores": d_move_sens}
        }
        return jsonify(resultado)

    except ValueError as ve:
        return jsonify({"erro": str(ve)}), 400
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)