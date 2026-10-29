# -*- coding: utf-8 -*-
"""История сессий: папки results/<источник>_<дата>_<время>/.

Для новых сессий итоги лежат в summary.json; для старых считаются по
analysis_log.csv (те же колонки, что пишет AnalysisLogger).
"""
import csv
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from app_paths import RESULTS_DIR

_NAME_RE = re.compile(r"^(?P<base>.+)_(?P<date>\d{8})_(?P<time>\d{6})(?:_\d+)?$")
SUMMARY_FILE = "summary.json"
LOG_FILE = "analysis_log.csv"
SUBTITLES_FILE = "subtitles.txt"


@dataclass
class SessionRecord:
    path: str
    name: str
    kind: str                 # webcam | screen | file
    source_label: str
    started: Optional[datetime]
    size_bytes: int = 0
    files: list = field(default_factory=list)

    @property
    def log_path(self):
        return os.path.join(self.path, LOG_FILE)

    @property
    def subtitles_path(self):
        return os.path.join(self.path, SUBTITLES_FILE)

    @property
    def has_log(self):
        return os.path.exists(self.log_path)


def _kind_and_label(base):
    if base == "webcam":
        return "webcam", "Веб-камера"
    if base == "screen":
        return "screen", "Экран"
    return "file", base


def list_sessions(results_dir=None):
    root = results_dir or RESULTS_DIR
    records = []
    if not os.path.isdir(root):
        return records
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        match = _NAME_RE.match(name)
        if match:
            base = match.group("base")
            try:
                started = datetime.strptime(match.group("date") + match.group("time"), "%Y%m%d%H%M%S")
            except ValueError:
                started = None
        else:
            base, started = name, None
        try:
            files = sorted(os.listdir(path))
            size = sum(os.path.getsize(os.path.join(path, f)) for f in files
                       if os.path.isfile(os.path.join(path, f)))
        except OSError:
            files, size = [], 0
        if not any(f in (LOG_FILE, SUBTITLES_FILE, SUMMARY_FILE) or f.endswith(".mp4") for f in files):
            continue
        if started is None:
            started = datetime.fromtimestamp(os.path.getmtime(path))
        kind, label = _kind_and_label(base)
        records.append(SessionRecord(path=path, name=name, kind=kind, source_label=label,
                                     started=started, size_bytes=size, files=files))
    records.sort(key=lambda r: r.started or datetime.min, reverse=True)
    return records


def _num(value):
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def summary_from_csv(path):
    """Итоги по журналу старого формата (без summary.json)."""
    with open(path, encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file, delimiter=";"))
    if not rows:
        return {}
    times = [_num(r.get("время_видео_с")) for r in rows]
    weights = []
    for i, t in enumerate(times):
        nxt = times[i + 1] if i + 1 < len(times) else None
        w = (nxt - t) if (t is not None and nxt is not None) else 0.7
        weights.append(min(max(w, 0.0), 3.0))

    total = sum(weights) or 1e-9
    visible = sum(w for w, r in zip(weights, rows) if r.get("лицо_найдено") == "да")
    duration = max((t for t in times if t is not None), default=0.0)

    blinks = [_num(r.get("моргания")) for r in rows if r.get("лицо_найдено") == "да"]
    blinks = [b for b in blinks if b is not None]
    blink_total = int(max(blinks) - min(blinks)) if blinks else 0
    blink_rate = blink_total / visible * 60.0 if visible >= 15.0 else None

    expressions = Counter()
    for w, r in zip(weights, rows):
        if r.get("лицо_найдено") == "да":
            expressions[r.get("мимика") or "?"] += w
    expr_total = sum(expressions.values())

    rapid, prev = 0, "нет"
    for r in rows:
        current = r.get("микровыражение") or "нет"
        if current != "нет" and current != prev:
            rapid += 1
        prev = current

    def values(col, require_face=True):
        out = []
        for r in rows:
            if require_face and r.get("лицо_найдено") != "да":
                continue
            v = _num(r.get(col))
            if v is not None:
                out.append(v)
        return out

    redness = values("покраснение_общее")
    deviation = [_num(r.get("напряжение_общее")) for r in rows
                 if r.get("лицо_найдено") == "да" and r.get("напряжение_статус") != "калибровка"]
    deviation = [v for v in deviation if v is not None]
    gaze_off = sum(w for w, r in zip(weights, rows)
                   if r.get("лицо_найдено") == "да" and r.get("взгляд") not in ("Центр", "Не найдено"))

    return {
        "duration_s": round(duration, 1),
        "visible_s": round(visible, 1),
        "face_share": round(visible / total, 3),
        "blink_total": blink_total,
        "blink_rate_mean": None if blink_rate is None else round(blink_rate, 1),
        "gaze_off_mean": round(gaze_off / visible, 3) if visible > 0 else None,
        "activity_mean": None,
        "redness_peak": round(max(redness), 3) if redness else None,
        "deviation_mean": round(sum(deviation) / len(deviation), 3) if deviation else None,
        "deviation_peak": round(max(deviation), 3) if deviation else None,
        "rapid_changes": rapid,
        "phrases": None,
        "expressions": [[name, round(share / expr_total, 3)]
                        for name, share in expressions.most_common(4)] if expr_total else [],
        "from_csv": True,
    }


def load_summary(record):
    path = os.path.join(record.path, SUMMARY_FILE)
    summary = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as file:
                summary = json.load(file)
        except Exception as error:
            print("Не удалось прочитать итоги сессии:", error)
    if not summary and record.has_log:
        try:
            summary = summary_from_csv(record.log_path)
        except Exception as error:
            print("Не удалось посчитать итоги по журналу:", error)
    if summary.get("phrases") is None and os.path.exists(record.subtitles_path):
        summary["phrases"] = sum(1 for line in read_lines(record.subtitles_path) if line.strip())
    return summary


def read_lines(path, limit=5000):
    try:
        with open(path, encoding="utf-8", errors="replace") as file:
            return [line.rstrip("\n") for _, line in zip(range(limit), file)]
    except OSError:
        return []


def read_transcript(record):
    return read_lines(record.subtitles_path)


def save_summary(session_dir, summary):
    if not session_dir:
        return None
    path = os.path.join(session_dir, SUMMARY_FILE)
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        return path
    except Exception as error:
        print("Не удалось сохранить итоги сессии:", error)
        return None


def delete_session(record, results_dir=None):
    """Удаляет папку сессии. Разрешено только внутри папки результатов."""
    root = os.path.realpath(results_dir or RESULTS_DIR)
    target = os.path.realpath(record.path)
    if os.path.dirname(target) != root:
        raise ValueError("Папка сессии находится вне папки результатов - удаление запрещено")
    shutil.rmtree(target)
