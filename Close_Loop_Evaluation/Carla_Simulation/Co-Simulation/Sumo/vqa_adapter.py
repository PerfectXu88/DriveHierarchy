# -*- coding: utf-8 -*-
"""
VQA data adapter for the Tongji CARLA+SUMO pipeline.

Generates the minimal ``vqa_data`` / ``anno`` dictionaries that
the close-loop VLM service expects, sourced from live CARLA state instead
of pre-recorded dataset files.

Runs inside **carla_base** (Python 3.8).
"""

import math

# ---------------------------------------------------------------------------
# Question templates – copied from B2DVL_Adapter/generator_modules/behaviour.py
# ---------------------------------------------------------------------------

QID42_QUESTION = (
    "Please predict the waypoint tokens for the next 4 seconds, "
    "with one set every 0.5 seconds, for a total of 8 sets of "
    "relative displacements."
)

QID50_QUESTION = (
    "Decide the immediate steering command and desired speed for the ego vehicle."
)

DEFAULT_ACTION_CONSIDERATIONS = (
    "When making the decision, consider the ego speed, the future route "
    "geometry, drivable space, nearby vehicles, pedestrians, traffic lights, "
    "traffic signs, and any immediate collision or off-road risk. Never idle "
    "without a concrete safety or rule-based reason. Use steering proactively "
    "to track the future path instead of defaulting to straight driving."
)

# Navigation command mapping (same as B2DVL)
COMMAND_MAP = {
    1: 'turn left at the intersection',
    2: 'turns right at the intersection',
    3: 'drive straight at the intersection',
    4: 'follow the road',
    5: 'do a lane change to the left',
    6: 'do a lane change to the right',
}


def _vec_len(v):
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def _distance_2d(loc_a, loc_b):
    return math.sqrt((loc_a.x - loc_b[0]) ** 2 + (loc_a.y - loc_b[1]) ** 2)


def _format_future_path_points(points):
    if not points:
        return "[]"
    return "[" + ", ".join(
        "(%.1f, %.1f)" % (float(point[0]), float(point[1]))
        for point in points
    ) + "]"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_anno_from_carla(hero_actor, nav_command=4,
                          future_path_points=None,
                          distance_to_destination=None):
    """Build a minimal ``anno`` dict from live CARLA hero state.

    Parameters
    ----------
    hero_actor : carla.Vehicle
        The ego vehicle actor.
    nav_command : int
        Navigation command integer (1-6).  Defaults to 4 (follow road).

    Returns
    -------
    dict
        A dict with the keys that ``generate_condition()`` reads:
        ``speed``, ``command_near``, ``command_far``,
        ``x``, ``y``, ``x_command_near``, ``y_command_near``,
        ``x_command_far``, ``y_command_far``.
    """
    vel = hero_actor.get_velocity()
    speed = _vec_len(vel)
    loc = hero_actor.get_location()

    return {
        'speed': speed,
        'command_near': nav_command,
        'command_far': nav_command,
        'x': loc.x,
        'y': loc.y,
        'x_command_near': loc.x,
        'y_command_near': loc.y,
        'x_command_far': loc.x,
        'y_command_far': loc.y,
        'future_path_points': list(future_path_points or []),
        'future_path_text': _format_future_path_points(future_path_points or []),
        'distance_to_destination': distance_to_destination,
    }


def generate_condition_from_anno(anno, qid):
    """Simplified version of B2DVL's ``generate_condition()``.

    Only handles qid 42 and 50 which are the two questions we ask
    during online CARLA inference.
    """
    condition = ""
    if qid == 42:
        condition += "The ego vehicle is driving at the speed of {:.1f} m/s. ".format(anno['speed'])
    elif qid == 50:
        condition += "The ego vehicle is driving at the speed of {:.1f} m/s, and it wants to ".format(anno['speed'])
        command_str = COMMAND_MAP.get(anno.get('command_near', 4), 'follow the road')
        condition += command_str + ". "
        if anno.get('distance_to_destination') is not None:
            condition += (
                "Remaining route distance to destination is approximately "
                "{:.1f} meters. ".format(float(anno['distance_to_destination']))
            )
        future_path_text = anno.get('future_path_text')
        if future_path_text:
            condition += (
                "The local future route in ego coordinates is given as "
                "(forward_m, right_m): %s. " % future_path_text
            )
    return condition


def build_action_decision_question(considerations_text=None,
                                   max_target_speed_mps=30.0):
    """Build an action-only prompt for online CARLA control."""
    considerations = considerations_text or DEFAULT_ACTION_CONSIDERATIONS
    considerations = considerations.strip()
    if considerations and not considerations.endswith(('.', '!', '?')):
        considerations += '.'

    return (
        "Your primary objective is to reach the destination in the minimum time "
        "while remaining safe and legal. "
        "Hard constraints: do not collide with vehicles, pedestrians, or static "
        "obstacles; do not leave the drivable route; obey traffic lights, stop "
        "signs, and right-of-way rules. "
        "Driving policy: do not idle, crawl, or wait without a concrete safety "
        "or rule-based reason; if the route ahead is clear, actively increase "
        "speed toward the highest safe speed; use steering proactively to follow "
        "the future path and begin steering before turns instead of defaulting "
        "to zero steering; slow down only when required by route geometry, "
        "traffic rules, obstacles, or collision risk. "
        "For the steer value, a negative value indicates left, while a positive value indicates right."
        + considerations + " "
        + "Output only the final action decision without explanation in the "
        + "exact format:\n"
        + "steer: <float in [-1.0, 1.0]>\n"
        + "target_speed_mps: <float in [0.0, %.1f]>" % float(max_target_speed_mps)
    )


def build_vqa_data(scenario_name, frame_number, hero_actor, nav_command=4):
    """Build the ``vqa_data`` dict expected by ``ask_single_frame()``.

    The returned structure mirrors what B2DVL's autopilot.py passes:

    .. code-block:: python

        {
            'scenario': str,
            'frame_number': int,
            'content': {
                'QA': {
                    'behaviour': [
                        {'qid': 42, 'Q': ..., 'A': ''},
                        {'qid': 50, 'Q': ..., 'A': ''},
                    ]
                },
                'key_object_infos': [],
                'extra_flags': {},
            },
            'anno': <anno_dict or path>,
        }
    """
    anno = build_anno_from_carla(hero_actor, nav_command)

    vqa_data = {
        'scenario': scenario_name,
        'frame_number': frame_number,
        'content': {
            'QA': {
                'behaviour': [
                    {'qid': 42, 'Q': QID42_QUESTION, 'A': ''},
                    {'qid': 50, 'Q': build_action_decision_question(), 'A': ''},
                ]
            },
            'key_object_infos': [],
            'extra_flags': {},
        },
        'anno': anno,  # pass dict directly; generate_condition_from_anno handles it
    }
    return vqa_data


def infer_nav_command_from_future_path(future_path_points):
    """Infer a coarse navigation command from local route points.

    ``future_path_points`` is a list of ``(forward_m, right_m)`` tuples in the
    ego coordinate frame.
    """
    if not future_path_points:
        return 4

    usable_points = [point for point in future_path_points if point[0] > 1.0]
    if not usable_points:
        return 4

    near_point = usable_points[min(len(usable_points) - 1, 1)]
    far_point = usable_points[-1]

    near_lateral = float(near_point[1])
    far_lateral = float(far_point[1])
    far_forward = max(float(far_point[0]), 1e-3)
    heading_deg = math.degrees(math.atan2(far_lateral, far_forward))

    if heading_deg <= -18.0 or near_lateral <= -4.0:
        return 1
    if heading_deg >= 18.0 or near_lateral >= 4.0:
        return 2
    if abs(heading_deg) <= 8.0 and abs(near_lateral) <= 1.5:
        return 3
    if near_lateral <= -2.0:
        return 5
    if near_lateral >= 2.0:
        return 6
    return 4
