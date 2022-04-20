# coding=utf-8
from turtle import color
import color_constance as cc
import numpy as np
import cv2

color_mapping = {
    "white": [
        "black",
        "deeppink1",
        "deeppink2",
        "deeppink3",
        "deeppink4",
        "deepskyblue1",
        "deepskyblue2",
        "deepskyblue3",
        "deepskyblue4",
        "darkgreen",
        "forestgreen",
        "mediumvioletred",
        "orangered1",
        "orangered2",
        "orangered3",
        "orangered4",
        "red1",
        "red2",
        "red3",
        "red4",
        "violetred",
        "violetred1",
        "violetred2",
        "violetred3",
        "violetre4",
    ],
    "green": [
        "white",
        "cadmiumorange",
        "darkorange",
        "orange",
        "orange1",
        "yellow1",
        "yellow2",
    ],
    "yellow1": ["black", "red1"],
    "red1": [
        "white",
        "purple",
        "purple1",
        "purple2",
        "purple3",
    ],
    "lightskyblue": ["dodgerblue4"],
    # black bg
    "black": [
        "white",
        "orangered1",
        "orangered2",
        "orangered3",
        "orangered4",
        "lightyellow1",
        "lightyellow2",
        "lightyellow3",
        "lightyellow4",
    ],
    "dodgerblue4": [
        "white",
        "orangered1",
        "orangered2",
        "orangered3",
        "orangered4",
        "lightyellow1",
        "lightyellow2",
        "lightyellow3",
        "lightyellow4",
    ],
    "darkgray": [
        "white",
        "lightyellow1",
        "lightyellow2",
        "lightyellow3",
        "lightyellow4",
    ],
    "red1": ["white"],
}

size = (1280, 960, 4)
for k in color_mapping:
    color_name = k
    color_value = cc.colors[color_name]
    img = np.zeros(size, dtype=np.uint8)
    r = color_value.red
    b = color_value.blue
    g = color_value.green
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    img[:, :, 3] = 255
    cv2.imwrite(f"{color_name}.png", img)
