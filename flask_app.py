import json
import pickle
import lightgbm as lgb
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# 対戦モードのJSONファイルを読み込む
with open("master.json", "r") as f:
    master = json.load(f)

# 学習済みのモデルをロードする
with open("win_ratio_model_lgb.pkl", mode="rb") as f:
    model = pickle.load(f)

weapon_hash = {}
for weapon in master["weapon"]:
    weapon_hash[weapon["value"]] = weapon


@app.route("/")
def index():

    return render_template(
        "index.html",
        master=master,
        lobby=1,
        power=0,
        mode=1,
        stage=1,
        a1_weapon=1,
        a2_weapon=1,
        a3_weapon=1,
        a4_weapon=1,
        b1_weapon=1,
        b2_weapon=1,
        b3_weapon=1,
        b4_weapon=1,
    )


@app.route("/submit", methods=["POST"])
def submit():
    # フォームの値を取得
    lobby = int(request.form.get("lobby"))
    power = int(request.form.get("power"))
    if lobby is not 4:  # XマッチではないならNanを入れる
        power = np.nan
    mode = int(request.form.get("mode"))
    stage = int(request.form.get("stage"))
    a1_weapon = int(request.form.get("a1_weapon"))
    a1_main = int(weapon_hash[a1_weapon]["main"])
    a1_sub = int(weapon_hash[a1_weapon]["sub"])
    a1_sp = int(weapon_hash[a1_weapon]["sp"])

    a2_weapon = int(request.form.get("a2_weapon"))
    a2_main = int(weapon_hash[a2_weapon]["main"])
    a2_sub = int(weapon_hash[a2_weapon]["sub"])
    a2_sp = int(weapon_hash[a2_weapon]["sp"])

    a3_weapon = int(request.form.get("a3_weapon"))
    a3_main = int(weapon_hash[a3_weapon]["main"])
    a3_sub = int(weapon_hash[a3_weapon]["sub"])
    a3_sp = int(weapon_hash[a3_weapon]["sp"])

    a4_weapon = int(request.form.get("a4_weapon"))
    a4_main = int(weapon_hash[a4_weapon]["main"])
    a4_sub = int(weapon_hash[a4_weapon]["sub"])
    a4_sp = int(weapon_hash[a4_weapon]["sp"])

    b1_weapon = int(request.form.get("b1_weapon"))
    b1_main = int(weapon_hash[b1_weapon]["main"])
    b1_sub = int(weapon_hash[b1_weapon]["sub"])
    b1_sp = int(weapon_hash[b1_weapon]["sp"])

    b2_weapon = int(request.form.get("b2_weapon"))
    b2_main = int(weapon_hash[b2_weapon]["main"])
    b2_sub = int(weapon_hash[b2_weapon]["sub"])
    b2_sp = int(weapon_hash[b2_weapon]["sp"])

    b3_weapon = int(request.form.get("b3_weapon"))
    b3_main = int(weapon_hash[b3_weapon]["main"])
    b3_sub = int(weapon_hash[b3_weapon]["sub"])
    b3_sp = int(weapon_hash[b3_weapon]["sp"])

    b4_weapon = int(request.form.get("b4_weapon"))
    b4_main = int(weapon_hash[b4_weapon]["main"])
    b4_sub = int(weapon_hash[b4_weapon]["sub"])
    b4_sp = int(weapon_hash[b4_weapon]["sp"])

    # 勝率予想に使うデータ
    inputData = np.array(
        [
            [
                lobby,
                mode,
                stage,
                power,
                a1_weapon,
                a1_main,
                a1_sub,
                a1_sp,
                a2_weapon,
                a2_main,
                a2_sub,
                a2_sp,
                a3_weapon,
                a3_main,
                a3_sub,
                a3_sp,
                a4_weapon,
                a4_main,
                a4_sub,
                a4_sp,
                b1_weapon,
                b1_main,
                b1_sub,
                b1_sp,
                b2_weapon,
                b2_main,
                b2_sub,
                b2_sp,
                b3_weapon,
                b3_main,
                b3_sub,
                b3_sp,
                b4_weapon,
                b4_main,
                b4_sub,
                b4_sp,
            ]
        ]
    )

    print(inputData)

    # 学習済みのモデルを使用して、予測を行う
    prob = 1 - model.predict(inputData)[0]

    # プルダウンのvalueの値をつなげた文字列を生成
    output = f"勝率{prob*100}%"

    return render_template(
        "index.html",
        master=master,
        lobby=lobby,
        power=power,
        mode=mode,
        stage=stage,
        a1_weapon=a1_weapon,
        a2_weapon=a2_weapon,
        a3_weapon=a3_weapon,
        a4_weapon=a4_weapon,
        b1_weapon=b1_weapon,
        b2_weapon=b2_weapon,
        b3_weapon=b3_weapon,
        b4_weapon=b4_weapon,
        output=output,
    )
