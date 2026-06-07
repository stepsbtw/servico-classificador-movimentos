from flask import Flask,request,jsonify
import torch,numpy as np,json
from pathlib import Path
from models import CNN1Conv

app=Flask(__name__)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR=Path("checkpoints")

with open("mapping.json") as f:
    m=json.load(f)
MAP_FALL={int(k):v[0] for k,v in m["y_classify_fall"].items()}
MAP_POSTURE={int(k):v[0] for k,v in m["y_classify_posture"].items()}
MAP_MOVEMENT={int(k):v[0] for k,v in m["y_classify_movement"].items()}

def carregar(tarefa,nome):
    n=2 if tarefa=="y_detect_fall" else len(MAP_FALL) if tarefa=="y_classify_fall" else len(MAP_POSTURE) if tarefa=="y_classify_posture" else len(MAP_MOVEMENT)
    model=CNN1Conv(8,num_classes=n).to(device)
    p=CHECKPOINT_DIR/tarefa/nome
    model.load_state_dict(torch.load(p/f"{nome}_FINAL.pth",map_location=device))
    model.eval()
    return model,np.load(p/f"{nome}_FINAL_mean.npy"),np.load(p/f"{nome}_FINAL_std.npy")

fall_det={s:carregar("y_detect_fall",s.upper()) for s in ["chest","left","right"]}
fall_cls={s:carregar("y_classify_fall",s.upper()) for s in ["chest","left","right"]}
posture_cls={s:carregar("y_classify_posture",s.upper()) for s in ["chest","left","right"]}
movement_cls={s:carregar("y_classify_movement",s.upper()) for s in ["chest","left","right"]}

def janela(d):
    acc=np.array(d["linear_acceleration"],dtype=np.float32)
    gyro=np.array(d["angular_speed"],dtype=np.float32)
    amag=np.sqrt((acc**2).sum(1,keepdims=True))
    wmag=np.sqrt((gyro**2).sum(1,keepdims=True))
    return np.concatenate((acc,amag,gyro,wmag),axis=1)

def infer(model,mean,std,x):
    x=(x-mean)/std
    x=torch.tensor(x.transpose(1,0)).unsqueeze(0).to(device)
    with torch.no_grad():
        p=torch.softmax(model(x),dim=1)
    return p.squeeze().cpu().numpy()

@app.route("/receber",methods=["POST"])
def receber():
    d=request.get_json()
    sensores=[s for s in ["chest","left","right"] if d.get(s)]
    out={}
    for nome,models in [("detect_fall",fall_det),("classify_fall",fall_cls),("classify_posture",posture_cls),("classify_movement",movement_cls)]:
        probs=None
        for s in sensores:
            p=infer(models[s][0],models[s][1],models[s][2],janela(d[s]))
            probs=p if probs is None else probs+p
        probs=probs/len(sensores)
        out[nome]={"classe":int(np.argmax(probs)),"probabilidades":probs.tolist()}
    return jsonify(out)

app.run(host="0.0.0.0",port=5000)
