#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from alifebook_lib.visualizers import MatrixVisualizer
from PIL import Image

#画像読み込み
img = Image.open('alife.png')
width, height = img.size

img_pixels = []
for x in range(height):
    for y in range(width):
        img_pixels.append(img.getpixel((y,x)))

img_pixels_norm = []
for i in range(height*width):
    p = img_pixels[i][0]
    if p == 0:
        img_pixels_norm.append(1.0)
    else:
        img_pixels_norm.append(0.)

img_pixels_norm = np.array( img_pixels_norm).reshape(height,width)


# visualizerの初期化 (Appendix参照)
visualizer = MatrixVisualizer()

# シミュレーションの各パラメタ
SPACE_GRID_SIZE = height
dx = 0.01
dt = 1
VISUALIZATION_STEP = 8  # 何ステップごとに画面を更新するか。

# モデルの各パラメタ
Du = 2e-5
Dv = 1e-5
f, k = 0.04, 0.06  # amorphous
#f, k = 0.035, 0.065  # spots
# f, k = 0.012, 0.05  # wandering bubbles
#f, k = 0.025, 0.05  # waves
#f, k = 0.022, 0.051 # stripe

# 初期化
u = np.ones((SPACE_GRID_SIZE, SPACE_GRID_SIZE))
u = u-img_pixels_norm*0.85
v = img_pixels_norm*0.25


# 中央にSQUARE_SIZE四方の正方形を置く
"""
SQUARE_SIZE = 20
u[SPACE_GRID_SIZE//2-SQUARE_SIZE//2:SPACE_GRID_SIZE//2+SQUARE_SIZE//2,
  SPACE_GRID_SIZE//2-SQUARE_SIZE//2:SPACE_GRID_SIZE//2+SQUARE_SIZE//2] = 0.5
v[SPACE_GRID_SIZE//2-SQUARE_SIZE//2:SPACE_GRID_SIZE//2+SQUARE_SIZE//2,
  SPACE_GRID_SIZE//2-SQUARE_SIZE//2:SPACE_GRID_SIZE//2+SQUARE_SIZE//2] = 0.25
"""
# 対称性を壊すために、少しノイズを入れる
u += np.random.rand(SPACE_GRID_SIZE, SPACE_GRID_SIZE)*0.1
v += np.random.rand(SPACE_GRID_SIZE, SPACE_GRID_SIZE)*0.1

while visualizer:  # visualizerはウィンドウが閉じられるとFalseを返す
    for i in range(VISUALIZATION_STEP):
        # ラプラシアンの計算
        laplacian_u = (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
                       np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4*u) / (dx*dx)
        laplacian_v = (np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) +
                       np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) - 4*v) / (dx*dx)
        # Gray-Scottモデル方程式
        dudt = Du*laplacian_u - u*v*v + f*(1.0-u)
        dvdt = Dv*laplacian_v + u*v*v - (f+k)*v
        u += dt * dudt
        v += dt * dvdt
    # 表示をアップデート
    visualizer.update(u)
