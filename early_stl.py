from flask import Flask,request,jsonify
import torch,numpy as np,json
from pathlib import Path
from models import CNN1Conv

app=Flask(__name__)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR=Path("checkpoints")

COMBINACOES={("chest","left","right"):("CHEST_LEFT_RIGHT",24),("chest","left"):("CHEST_LEFT",16),("chest","right"):("CHEST_RIGHT",16),("left","right"):("LEFT_RIGHT",16),("chest",):("CHEST",8),("left",):("LEFT",8),("right",):("RIGHT",8)}

with open("mapping.json") as f:
    m=json.load(f)
MAP_FALL={int(k):v[0] for k,v in m["y_classify_fall"].items()}
MAP_POSTURE={int(k):v[0] for k,v in m["y_classify_posture"].items()}
MAP_MOVEMENT={int(k):v[0] for k,v in m["y_classify_movement"].items()}

def carregar(tarefa,nome,n_feat):
    n=2 if tarefa=="y_detect_fall" else len(MAP_FALL) if tarefa=="y_classify_fall" else len(MAP_POSTURE) if tarefa=="y_classify_posture" else len(MAP_MOVEMENT)
    model=CNN1Conv(n_feat,num_classes=n).to(device)
    p=CHECKPOINT_DIR/tarefa/nome
    model.load_state_dict(torch.load(p/f"{nome}_FINAL.pth",map_location=device))
    model.eval()
    return model,np.load(p/f"{nome}_FINAL_mean.npy"),np.load(p/f"{nome}_FINAL_std.npy")

fall_det,fall_cls,posture_cls,movement_cls={}, {}, {}, {}

for combo,(nome,n_feat) in COMBINACOES.items():
    fall_det[combo]=carregar("y_detect_fall",nome,n_feat)
    fall_cls[combo]=carregar("y_classify_fall",nome,n_feat)
    posture_cls[combo]=carregar("y_classify_posture",nome,n_feat)
    movement_cls[combo]=carregar("y_classify_movement",nome,n_feat)

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
    return int(torch.argmax(p,1)),p.squeeze().cpu().tolist()

@app.route("/receber",methods=["POST"])
def receber():
    d=request.get_json()
    sensores=[s for s in ["chest","left","right"] if d.get(s)]
    combo=tuple(sensores)
    x=np.concatenate([janela(d[s]) for s in sensores],axis=1)
    out={}
    for nome,models in [("detect_fall",fall_det),("classify_fall",fall_cls),("classify_posture",posture_cls),("classify_movement",movement_cls)]:
        c,p=infer(models[combo][0],models[combo][1],models[combo][2],x)
        out[nome]={"classe":c,"probabilidades":p}
    return jsonify(out)

app.run(host="0.0.0.0",port=5000)