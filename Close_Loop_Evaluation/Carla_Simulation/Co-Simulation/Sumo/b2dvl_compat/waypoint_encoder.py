# -*- coding: utf-8 -*-
"""
Verbatim copy of B2DVL_Adapter/waypoint_encoder.py
Pure Python, no external dependencies.  Python 3.8 compatible.
"""

import math

X_TOKEN_PREFIX = 'x'
X_TOKEN_POS = 'fwd'
X_TOKEN_NEG = 'back'
X_TOKEN_ZERO = 'stay'
Y_TOKEN_PREFIX = 'y'
Y_TOKEN_POS = 'right'
Y_TOKEN_NEG = 'left'
Y_TOKEN_ZERO = 'stay'
DIR_TOKEN_PREFIX = 'dir'
DIR_TOKEN_POS = 'right'
DIR_TOKEN_NEG = 'left'
DIR_TOKEN_ZERO = 'fwd'
DIR_TOKEN_BACK = 'back'
SPD_TOKEN_PREFIX = 'spd'


def float_to_token(val):
    abs_val = abs(val)
    if abs_val >= 15 + 15 / 16:
        return 'ff'
    token = round(abs_val * 16)
    return '{:02x}'.format(token)


def generate_motion_tokens(points):
    motion_tokens = []
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]

        x_token = float_to_token(x)
        if x_token == '00':
            x_token = "<{}_{}_00>".format(X_TOKEN_PREFIX, X_TOKEN_ZERO)
        elif x > 0:
            x_token = "<{}_{}_{}>".format(X_TOKEN_PREFIX, X_TOKEN_POS, x_token)
        else:
            x_token = "<{}_{}_{}>".format(X_TOKEN_PREFIX, X_TOKEN_NEG, x_token)

        y_token = float_to_token(y)
        if y_token == '00':
            y_token = "<{}_{}_00>".format(Y_TOKEN_PREFIX, Y_TOKEN_ZERO)
        elif y > 0:
            y_token = "<{}_{}_{}>".format(Y_TOKEN_PREFIX, Y_TOKEN_POS, y_token)
        else:
            y_token = "<{}_{}_{}>".format(Y_TOKEN_PREFIX, Y_TOKEN_NEG, y_token)

        motion_tokens.append(x_token)
        motion_tokens.append(y_token)
    return motion_tokens


def angle_to_token(angle):
    if angle < -180:
        angle += 360
    elif angle >= 180:
        angle -= 360
    abs_angle = abs(angle)
    if abs_angle <= 180:
        token = round(abs_angle / 180 * 64)
    else:
        token = 64
    if token == 0:
        return "<{}_{}_00>".format(DIR_TOKEN_PREFIX, DIR_TOKEN_ZERO)
    elif token == 64:
        return "<{}_{}_40>".format(DIR_TOKEN_PREFIX, DIR_TOKEN_BACK)
    direction = DIR_TOKEN_POS if angle > 0 else DIR_TOKEN_NEG
    return "<{}_{}_{:02x}>".format(DIR_TOKEN_PREFIX, direction, token)


def distance_to_token(distance):
    if distance >= 15 + 15 / 16:
        return "<{}_ff>".format(SPD_TOKEN_PREFIX)
    token = round(distance * 16)
    return "<{}_{:02x}>".format(SPD_TOKEN_PREFIX, token)


def generate_motion_and_direction_tokens(points):
    motion_tokens = []
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        angle = math.degrees(math.atan2(y, x))
        distance = math.sqrt(x ** 2 + y ** 2)
        dir_token = angle_to_token(angle)
        spd_token = distance_to_token(distance)
        if "00" in spd_token:
            dir_token = "<{}_{}_00>".format(DIR_TOKEN_PREFIX, DIR_TOKEN_ZERO)
        motion_tokens.append(dir_token)
        motion_tokens.append(spd_token)
    return motion_tokens
