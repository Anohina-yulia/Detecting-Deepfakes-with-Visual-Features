# -*- coding: utf-8 -*-
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import json
import traceback

from deepface import DeepFace


def normalize_emotion_name(name):
    mapping = {
        "angry": "злость",
        "disgust": "отвращение",
        "fear": "страх",
        "happy": "радость",
        "sad": "грусть",
        "surprise": "удивление",
        "neutral": "нейтрально",
    }
    return mapping.get(str(name).lower(), str(name))


def analyze_emotion(image_path):
    result = DeepFace.analyze(
        img_path=image_path,
        actions=["emotion"],
        enforce_detection=False,
        silent=True
    )

    if isinstance(result, list):
        result = result[0] if result else {}

    emotion_scores = result.get("emotion", {})
    dominant = result.get("dominant_emotion", "neutral")

    confidence = 0.0
    if emotion_scores:
        confidence = float(max(emotion_scores.values()))

    return {
        "ok": True,
        "dominant_emotion": normalize_emotion_name(dominant),
        "confidence": confidence,
        "scores": emotion_scores,
    }


def main():
    try:
        if len(sys.argv) < 2:
            print(json.dumps({
                "ok": False,
                "error": "Image path argument is missing"
            }, ensure_ascii=False))
            return

        image_path = sys.argv[1]

        if not os.path.exists(image_path):
            print(json.dumps({
                "ok": False,
                "error": f"Image file not found: {image_path}"
            }, ensure_ascii=False))
            return

        result = analyze_emotion(image_path)
        print(json.dumps(result, ensure_ascii=False))

    except Exception as error:
        print(json.dumps({
            "ok": False,
            "error": str(error),
            "traceback": traceback.format_exc(limit=1)
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()