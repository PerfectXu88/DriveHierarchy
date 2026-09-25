# -*- coding: utf-8 -*-
"""
Minimal copy of SpeedCommand / DirectionCommand from
B2DVL_Adapter/generator_modules/behaviour.py

Plus helper functions to parse VLM text output into command enums.
Python 3.8 compatible.
"""

import re


class SpeedCommand:
    keep = 'KEEP'
    accelerate = 'ACCELERATE'
    decelerate = 'DECELERATE'
    stop = 'STOP'

    ALL = ('KEEP', 'ACCELERATE', 'DECELERATE', 'STOP')


class DirectionCommand:
    follow = 'FOLLOW_LANE'
    left_change = 'CHANGE_LANE_LEFT'
    right_change = 'CHANGE_LANE_RIGHT'
    straight = 'GO_STRAIGHT'
    left_turn = 'TURN_LEFT'
    right_turn = 'TURN_RIGHT'
    left_deviate = 'DEVIATE_LEFT'
    right_deviate = 'DEVIATE_RIGHT'

    ALL = (
        'FOLLOW_LANE', 'CHANGE_LANE_LEFT', 'CHANGE_LANE_RIGHT',
        'GO_STRAIGHT', 'TURN_LEFT', 'TURN_RIGHT',
        'DEVIATE_LEFT', 'DEVIATE_RIGHT',
    )


def extract_keys(text):
    """Extract a single (direction, speed) pair from VLM answer text.

    Expected format examples:
        "FOLLOW_LANE, KEEP"
        "Direction Key = FOLLOW_LANE, Speed Key = KEEP"
    """
    dir_cmd = None
    spd_cmd = None
    for d in DirectionCommand.ALL:
        if d in text:
            dir_cmd = d
            break
    for s in SpeedCommand.ALL:
        if s in text:
            spd_cmd = s
            break
    return dir_cmd, spd_cmd


def extract_key_list(text):
    """Extract lists of (direction, speed) commands from VLM answer text.

    The VLM may output multiple command pairs; this returns two lists.
    """
    dir_cmds = []
    spd_cmds = []
    for d in DirectionCommand.ALL:
        if d in text:
            dir_cmds.append(d)
    for s in SpeedCommand.ALL:
        if s in text:
            spd_cmds.append(s)
    return dir_cmds, spd_cmds


CONTROL_VALUE_RE = re.compile(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)')


def _extract_float_after_label(text, labels):
    lower_text = text.lower()
    for label in labels:
        pattern = r'%s\s*[:=]\s*(%s)' % (re.escape(label.lower()), CONTROL_VALUE_RE.pattern)
        match = re.search(pattern, lower_text)
        if match:
            return float(match.group(1))
    return None


def extract_control_values(text):
    """Extract ``steer`` and ``target_speed_mps`` from VLM output text."""
    if not text:
        return None, None

    steer = _extract_float_after_label(text, ('steer',))
    target_speed_mps = _extract_float_after_label(
        text,
        ('target_speed_mps', 'target_speed'),
    )
    return steer, target_speed_mps
