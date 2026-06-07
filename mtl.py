from typing import Dict
from flask import Flask, request, jsonify
import torch
import numpy as np
from pathlib import Path
from models import MultiTaskModel

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints")

COMBINACOES_EARLY = {
    ("chest", "left", "right"): ("CHEST_LEFT_RIGHT", 24),
    ("chest", "left"): ("CHEST_LEFT", 16),
    ("chest", "right"): ("CHEST_RIGHT", 16),
    ("left", "right"): ("LEFT_RIGHT", 16),
    ("chest",): ("CHEST", 8),
    ("left",): ("LEFT", 8),
    ("right",): ("RIGHT", 8)
}

def carregar_modelo(nome: str, n_feat: int):
    modelo = MultiTaskModel(n_features=n_feat).to(device)
    caminho = CHECKPOINT_DIR / nome
    modelo.load_state_dict(torch.load(caminho / f"{nome}_FINAL.pth", map_location=device))
    modelo.eval()
    mean = np.load(caminho / f"{nome}_FINAL_mean.npy")
    std = np.load(caminho / f"{nome}_FINAL_std.npy")
    return modelo, mean, std

modelos = {combo: carregar_modelo(nome, n_feat) for combo, (nome, n_feat) in COMBINACOES_EARLY.items()}

def construir_janela(dados: Dict):
    acc = np.array(dados["linear_acceleration"], dtype=np.float32)
    gyro = np.array(dados["angular_speed"], dtype=np.float32)
    amag = np.sqrt(np.sum(acc ** 2, axis=1, keepdims=True))
    wmag = np.sqrt(np.sum(gyro ** 2, axis=1, keepdims=True))
    return np.concatenate((acc, amag, gyro, wmag), axis=1)

def inferencia(modelo, mean, std, janela):
    x = (janela - mean) / std
    x = torch.tensor(x.transpose(1, 0), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        fall_det, fall_cls, posture, movement = modelo(x)

    p_fall_det = torch.softmax(fall_det, dim=1)
    p_fall_cls = torch.softmax(fall_cls, dim=1)
    p_posture = torch.softmax(posture, dim=1)
    p_movement = torch.softmax(movement, dim=1)

    return {
        "detect_fall": {
            "classe": int(torch.argmax(p_fall_det, dim=1)),
            "probabilidades": p_fall_det.squeeze().cpu().tolist()
        },
        "classify_fall": {
            "classe": int(torch.argmax(p_fall_cls, dim=1)),
            "probabilidades": p_fall_cls.squeeze().cpu().tolist()
        },
        "classify_posture": {
            "classe": int(torch.argmax(p_posture, dim=1)),
            "probabilidades": p_posture.squeeze().cpu().tolist()
        },
        "classify_movement": {
            "classe": int(torch.argmax(p_movement, dim=1)),
            "probabilidades": p_movement.squeeze().cpu().tolist()
        }
    }

@app.route("/receber", methods=["POST"])
def receber():
    data = request.get_json()

    sensores = [s for s in ["chest", "left", "right"] if data.get(s)]

    if not sensores:
        return jsonify({"erro": "Nenhum sensor enviado"}), 400

    combo = tuple(sensores)

    if combo not in modelos:
        return jsonify({"erro": "Combinação inválida"}), 400

    janelas = [construir_janela(data[s]) for s in sensores]
    janela_fusa = np.concatenate(janelas, axis=1)

    modelo, mean, std = modelos[combo]

    return jsonify(inferencia(modelo, mean, std, janela_fusa))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)