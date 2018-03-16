#=========================================
#
# handson_kerasで学習したモデルを使って
# 自分で作成した手書き文字を認識するプログラム
# (精度が十分でません。kerasデータセットの手書き文字画像は
#  中央にセンタリングされています。なので、手書き文字を書くときは
#  中央に寄せて書いてください。。)
#
# フォルダ構成
#
# keras_predict.py
# model.json
# weights.hdf5
# image
#  |- *.jpg (ペイントなどで書いた手書き数字をここにいれてね)
#=========================================

import keras
from keras.models import model_from_json
from keras.optimizers import RMSprop

from PIL import Image # pip install Pillowが必要
import numpy as np
import glob # pip install globが必要(かも)

model_filename = "model.json"
weight_filename = "weights.hdf5"

#========================
# 学習済NNのロード
#========================
# モデルのロード
model = model_from_json(open(model_filename).read())

# 学習済weight,biasのロード
model.load_weights(weight_filename)

# モデルのコンパイル
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#========================
# 画像の取込み
#========================
files = glob.glob("image/*.jpg")

for filename in files:
    # 画像を読み込み
    img = Image.open(filename)

    # 学習モデルにあわせて、画像を28*28に圧縮
    img_resize = img.resize((28, 28))

    # 学習モデルにあわせて、画像をグレースケールに
    img_gray = img_resize.convert('L')

    # 数値に変換
    data = np.array(img_gray).reshape(1, 784)

    # 黒ぬき画像→白ぬき画像に変換
    data = 255 - data

    #========================
    # 画像の推論
    #========================
    result = model.predict_proba(data)
    print("filename={}, result={}".format(filename,result))
