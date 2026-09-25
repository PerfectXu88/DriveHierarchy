# -*- coding: utf-8 -*-
"""
Minimal copied inference utilities
(Bubble, Context, create_query, create_response).

Kept wire-compatible with the VLM service (/interact endpoint).
Python 3.8 compatible.
"""

from datetime import datetime


class Context:
    def __init__(self, conversation_window, no_history):
        self.conversation_window = conversation_window
        self.no_history = no_history

        self.conversation = []
        self.frames = []
        self.reset()

    def reset(self):
        self.conversation = []

    def update(self, new_bubble):
        if self.no_history is True:
            return
        self.conversation.append(new_bubble)
        if new_bubble.frame_number not in self.frames:
            self.frames.append(new_bubble.frame_number)

    def fifo(self):
        if len(self.frames) >= self.conversation_window:
            self.clean_old_bubbles(self.frames[-self.conversation_window])

    def clean_old_bubbles(self, threshold):
        self.conversation[:] = [x for x in self.conversation if x.frame_number > threshold]

    def get_context_for_question(self, qid, prev, inherit, frame_number):
        ret_context = [
            x for x in self.conversation
            if (x.frame_number == frame_number and x.qid in prev.get(qid, []))
            or (x.frame_number < frame_number and x.qid in inherit.get(qid, []))
        ]
        return ret_context

    def __str__(self):
        s = "Context(\n"
        for bubble in self.conversation:
            s += "    [{}, frame {}] {}: {}\n".format(
                bubble.scenario, bubble.frame_number, bubble.actor, bubble.get_full_words())
            s += "    [{}, frame {}] images: {}, extra_images: {}\n".format(
                bubble.scenario, bubble.frame_number, bubble.images, bubble.extra_images)
        s += ")"
        return s


def create_query(words, images, frame_number, scenario, qid, gt,
                 transform=None, extra_words=None, extra_images=None):
    if extra_images is None:
        extra_images = []
    new_bubble = Bubble(
        actor='User', words=words, images=images,
        frame_number=frame_number, scenario=scenario,
        extra_words=extra_words, extra_images=extra_images,
        qid=qid, gt=gt, timestamp=datetime.now().timestamp(),
        transform=transform
    )
    return new_bubble


def create_response(words, frame_number, scenario, qid, gt):
    new_bubble = Bubble(
        actor='VLM', words=words, images=[],
        frame_number=frame_number, scenario=scenario,
        extra_words=None, extra_images=[],
        qid=qid, gt=gt, timestamp=datetime.now().timestamp(),
        transform=None
    )
    return new_bubble


class Bubble:
    def __init__(self, actor, words, images, frame_number, scenario,
                 transform=None, extra_words=None, extra_images=None,
                 qid=-1, gt=None, timestamp=None):
        self.actor = actor
        self.words = words
        self.images = images if images is not None else []
        self.frame_number = frame_number
        self.scenario = scenario
        self.extra_words = extra_words
        self.extra_images = extra_images if extra_images is not None else []
        self.qid = qid
        self.gt = gt
        self.timestamp = timestamp
        self.transform = transform

    def get_full_words(self):
        if self.extra_words is not None:
            return self.extra_words + self.words
        else:
            return self.words

    def get_full_images(self):
        image_dict = {}
        for images in self.images:
            image_dict[images['frame_number']] = {}
            for key, value in images.items():
                if key != 'frame_number':
                    image_dict[images['frame_number']][key] = value
        for images in self.extra_images:
            image_dict[images['frame_number']] = {}
            for key, value in images.items():
                if key != 'frame_number':
                    image_dict[images['frame_number']][key] = value
        return image_dict

    def __str__(self):
        s = "Bubble(\n"
        s += "    qid = {}\n".format(self.qid)
        s += "    actor = {}\n".format(self.actor)
        s += "    full_words = {}\n".format(self.get_full_words())
        s += "    full_images = {}\n".format(self.get_full_images())
        s += "    frame_number = {}\n".format(self.frame_number)
        s += "    scenario = {}\n".format(self.scenario)
        s += "    gt = {}\n".format(self.gt)
        s += ")"
        return s

    def to_dict(self):
        return {
            "actor": self.actor,
            "words": self.words,
            "images": self.images,
            "frame_number": self.frame_number,
            "scenario": self.scenario,
            "extra_words": self.extra_words,
            "extra_images": self.extra_images,
            "qid": self.qid,
            "gt": self.gt,
            "timestamp": self.timestamp,
            "transform": self.transform
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            actor=data["actor"],
            words=data["words"],
            images=data.get("images", []),
            frame_number=data["frame_number"],
            scenario=data["scenario"],
            transform=data.get("transform"),
            extra_words=data.get("extra_words"),
            extra_images=data.get("extra_images", []),
            qid=data.get("qid", -1),
            gt=data.get("gt"),
            timestamp=data.get("timestamp")
        )
