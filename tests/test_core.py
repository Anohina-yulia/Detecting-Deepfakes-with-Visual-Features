# -*- coding: utf-8 -*-
"""Проверки ядра без интерфейса.

Запуск из папки проекта:
    .venv_main\\Scripts\\python.exe -m unittest discover -s tests -v
"""
import csv
import os
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analysis_logger import HEADER  # noqa: E402
from app_paths import FACE_MODEL_PATH  # noqa: E402
from eye_tracker import EyeTrackerModule  # noqa: E402
from models import EV_MICRO, EV_SPEECH, SessionEvent  # noqa: E402
from redness_analyzer import RednessAnalyzerModule  # noqa: E402
from session_history import SessionRecord, delete_session, summary_from_csv  # noqa: E402
from session_timeline import SessionTimeline  # noqa: E402
from settings_store import SettingsStore  # noqa: E402
from speech_engine import SpeechSession, Word, agreed_prefix  # noqa: E402
from text_correction import SafeMeaningCorrectionModule  # noqa: E402
from whisper_subtitle import WhisperSubtitleModule  # noqa: E402

TEST_VIDEO = os.path.join(ROOT, "results", "IMG_1761_analyzed.mp4")

# "истинная" речь: (слово, начало, конец) в секундах
SCRIPT = [("Здравствуйте!", 0.2, 1.0), ("Сегодня", 2.0, 2.4), ("мы", 2.45, 2.8),
          ("проверяем", 2.8, 3.5), ("программу", 3.5, 4.2), ("анализа", 4.2, 4.8),
          ("видео.", 4.8, 5.15), ("Как", 6.2, 6.5), ("прошёл", 6.5, 6.9), ("ваш", 6.9, 7.1),
          ("день?", 7.1, 7.5)]


def _time_coded_chunk(start, size, sr):
    """В образцах звука закодирован их абсолютный номер - по нему поддельная модель
    понимает, какой отрезок речи ей передали."""
    return (np.arange(start, start + size, dtype=np.float64) / 1e6).astype(np.float32)


class _FakeModel:
    """Имитирует Whisper: целые слова внутри буфера, обрывки на его краях."""

    sr = 16000

    def transcribe(self, audio, **kwargs):
        a = float(audio[0]) * 1e6 / self.sr
        b = a + len(audio) / self.sr
        words = []
        for text, s, e in SCRIPT:
            if s >= a - 0.05 and e <= b:
                words.append((text, s, e))
            elif s < b < e and b - s > 0.12:  # окно обрезало слово справа
                words.append((text[: max(2, len(text) // 2)] + "...", s, b))
            elif s < a < e and e - a > 0.12:  # и слева
                words.append((text[len(text) // 2:], a, e))
        segment = SimpleNamespace(
            text=" ".join(w[0] for w in words), no_speech_prob=0.01, avg_logprob=-0.2,
            words=[SimpleNamespace(word=" " + t, start=s - a, end=e - a) for t, s, e in words])
        return iter([segment] if words else []), None


class _FakeStorage:
    def __init__(self):
        self.final, self.live = [], []

    def append_final(self, text, t=None):
        self.final.append(text)

    def append_live(self, text):
        self.live.append(text)


def _fake_session():
    whisper = WhisperSubtitleModule.__new__(WhisperSubtitleModule)  # без загрузки модели
    whisper.lock = threading.Lock()
    whisper.reset()
    whisper.model = _FakeModel()
    models = SimpleNamespace(whisper=whisper,
                             corrector=SafeMeaningCorrectionModule(load_model=False))
    audio = SimpleNamespace(samplerate=16000)
    storage, events = _FakeStorage(), []
    holder = {}
    session = SpeechSession(models, audio, storage, time_fn=lambda: holder["s"].audio_time,
                            on_event=events.append)
    holder["s"] = session
    return session, storage, events


class SpeechTextTests(unittest.TestCase):
    def test_hallucinations_removed(self):
        strip = WhisperSubtitleModule.strip_hallucinations
        self.assertEqual(strip("Субтитры А.Семкин."), "")
        self.assertEqual(strip("Субтитры субтитров Н.Новикова"), "")
        self.assertEqual(strip("Субтитры сделал DimaTorzok"), "")
        self.assertEqual(strip("Продолжение следует..."), "")
        self.assertEqual(strip("Привет, как дела? Субтитры создавал DimaTorzok."), "Привет, как дела?")

    def test_real_speech_kept(self):
        strip = WhisperSubtitleModule.strip_hallucinations
        text = "Включи, пожалуйста, субтитры на английском языке в этом фильме."
        self.assertEqual(strip(text), text)
        self.assertEqual(strip("Спасибо за просмотр фильма вместе с нами"),
                         "Спасибо за просмотр фильма вместе с нами")

    def test_agreed_prefix(self):
        a = [Word(0, 1, "Сегодня"), Word(1, 2, "мы"), Word(2, 3, "про...")]
        b = [Word(0, 1, "сегодня,"), Word(1, 2, "мы"), Word(2, 3, "проверяем")]
        self.assertEqual([w.text for w in agreed_prefix(a, b)], ["сегодня,", "мы"])
        self.assertEqual(agreed_prefix([], b), [])

    def test_streaming_without_fragments_or_repeats(self):
        """Окно режет слова с обоих краёв, но в субтитры попадают только целые фразы."""
        session, storage, events = _fake_session()
        sr = 16000
        for start in range(0, sr * 9, 1024):  # 9 с звука блоками, как с микрофона
            session.feed(_time_coded_chunk(start, 1024, sr))
        session._flush()
        self.assertEqual(storage.final, ["Здравствуйте!",
                                         "Сегодня мы проверяем программу анализа видео.",
                                         "Как прошёл ваш день?"])
        self.assertEqual([e.text for e in events], storage.final)
        for event, expected in zip(events, (0.2, 2.0, 6.2)):  # время начала фразы, а не фиксации
            self.assertAlmostEqual(event.t, expected, delta=0.3)
        raw = " ".join(text for _, text in session.drain_speech())
        self.assertEqual(raw.count("ваш"), 1)

    def test_new_tail_overlap(self):
        tail = WhisperSubtitleModule.new_tail
        whisper = WhisperSubtitleModule.__new__(WhisperSubtitleModule)  # без загрузки модели
        self.assertEqual(tail(whisper, "Здравствуйте! Сегодня мы",
                              "Сегодня мы проверяем программу"), "проверяем программу")
        self.assertEqual(tail(whisper, "один два три", "два три"), "")


class Snapshot:
    """Минимальный снимок для проверки временной шкалы."""

    def __init__(self, t, face=True, activity=0.2, blink_rate=15.0):
        self.t = t
        self.face_detected = face
        self.activity = activity if face else None
        self.blink_rate = blink_rate if face else None
        self.gaze_off_ratio = 0.1 if face else None
        self.redness = None
        self.redness_ready = False
        self.deviation = None
        self.expression = "Нейтрально"
        self.expression_intensity = 0.1
        self.blink_count = int(t / 4)
        self.visible_time = t
        self.baseline_blink_rate = 17.0
        self.baseline_gaze_off = None
        self.baseline_activity = None


class TimelineTests(unittest.TestCase):
    def test_sampling_and_summary(self):
        tl = SessionTimeline()
        for i in range(400):  # 40 с по 0.1 с
            t = i * 0.1
            tl.add_snapshot(Snapshot(t, face=not (10.0 <= t < 12.0)))
        tl.add_event(SessionEvent(5.0, EV_MICRO, "краткая улыбка"))
        tl.add_event(SessionEvent(6.0, EV_SPEECH, "Привет"))
        self.assertTrue(150 <= len(tl.t) <= 165, len(tl.t))
        self.assertTrue(all(b > a for a, b in zip(tl.t, tl.t[1:])))
        summary = tl.summary()
        self.assertEqual(summary["rapid_changes"], 1)
        self.assertEqual(summary["phrases"], 1)
        self.assertAlmostEqual(summary["face_share"], 0.95, delta=0.03)
        self.assertIsNone(tl.value_at("activity", 11.0))  # лица не было
        self.assertAlmostEqual(tl.value_at("activity", 20.0), 0.2)

    def test_bucketing_keeps_range(self):
        tl = SessionTimeline()
        for i in range(4000):
            tl.add_snapshot(Snapshot(i * 0.3))
        times, values = tl.bucketed("activity", 0, tl.t[-1], 100)
        self.assertEqual(len(times), 100)
        self.assertTrue(all(abs(v - 0.2) < 1e-9 for v in values if v is not None))


class EyeTests(unittest.TestCase):
    def test_closure_duration_and_blink(self):
        eyes = EyeTrackerModule()
        t = 0.0
        for ear in [0.32] * 10 + [0.05] * 6 + [0.33] * 10:
            eyes.update_blink_counter(ear, now=t)
            t += 1 / 30
        self.assertEqual(eyes.blink_count, 1)
        self.assertIsNotNone(eyes.last_closure_ms)
        self.assertTrue(80 <= eyes.last_closure_ms <= 400, eyes.last_closure_ms)


class RednessTests(unittest.TestCase):
    def test_skin_survives_colored_light(self):
        """Кожа под фиолетовым светом: нормализация по кадру сохраняет её, по участку - нет."""
        frame = np.zeros((360, 640, 3), np.uint8)
        frame[:] = (150, 90, 120)                     # фиолетовый фон (BGR)
        skin = np.zeros((40, 56, 3), np.uint8)
        skin[:] = (120, 135, 205)                     # кожа под тем же светом
        rng = np.random.default_rng(0)
        skin = np.clip(skin.astype(int) + rng.integers(-6, 7, skin.shape), 0, 255).astype(np.uint8)
        frame[100:140, 100:156] = skin
        ra = RednessAnalyzerModule()
        gains = ra.gray_world_gains(frame)
        self.assertGreater(ra.compute_patch_metric(skin, gains), 0.0)
        self.assertEqual(ra.compute_patch_metric(ra.gray_world_normalize(skin)), 0.0)


@unittest.skipUnless(os.path.exists(FACE_MODEL_PATH), "нет модели лица")
class PipelineTests(unittest.TestCase):
    def test_no_face_frames(self):
        from analysis_pipeline import AnalysisPipeline

        pl = AnalysisPipeline(FACE_MODEL_PATH)
        try:
            frame = np.zeros((720, 1280, 3), np.uint8)
            for i in range(30):
                s = pl.process(frame, i / 30)
            self.assertTrue(s.face_model_ok)
            self.assertFalse(s.face_detected)
            for value in (s.blink_rate, s.activity, s.gaze_off_ratio, s.expression_intensity):
                self.assertIsNone(value)
            self.assertFalse(s.redness_ready)
        finally:
            pl.close()

    def test_missing_model_does_not_crash(self):
        from analysis_pipeline import AnalysisPipeline

        pl = AnalysisPipeline(os.path.join(tempfile.gettempdir(), "нет_такой_модели.task"))
        self.assertFalse(pl.face_available)
        s = pl.process(np.zeros((720, 1280, 3), np.uint8), 0.0)
        self.assertFalse(s.face_model_ok)
        self.assertFalse(s.face_detected)

    @unittest.skipUnless(os.path.exists(TEST_VIDEO), "нет тестового видео")
    def test_video_calibrates(self):
        from analysis_pipeline import AnalysisPipeline
        from video_source import VideoFileSource

        pl = AnalysisPipeline(FACE_MODEL_PATH)
        src = VideoFileSource(TEST_VIDEO)
        try:
            snapshot = None
            while True:  # всё видео (16.6 с): частота морганий появляется после 15 с с лицом
                ok, frame = src.read()
                if not ok:
                    break
                snapshot = pl.process(frame, src.get_time())
            self.assertTrue(snapshot.face_detected)
            self.assertTrue(snapshot.mimic_ready)
            self.assertTrue(snapshot.redness_ready, "покраснение должно откалиброваться")
            self.assertIsNotNone(snapshot.activity)
            self.assertTrue(0.0 <= snapshot.activity <= 1.0)
            self.assertIsNotNone(snapshot.blink_rate)
            kinds = {e.kind for e in pl.drain_events()}
            self.assertIn("калибровка", kinds)
            overlay = pl.build_overlay()
            self.assertIsNotNone(overlay.bbox)
            self.assertEqual(len(overlay.left_eye), 16)
        finally:
            src.release()
            pl.close()


class StorageTests(unittest.TestCase):
    def test_settings_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "settings.json")
            store = SettingsStore(path)
            store.set("camera_index", 2)
            store.set("unknown_key", 1)
            store.save()
            again = SettingsStore(path)
            self.assertEqual(again["camera_index"], 2)
            self.assertTrue(again["save_log"])
            self.assertNotIn("unknown_key", again.data)

    def test_summary_from_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "analysis_log.csv")
            with open(path, "w", newline="", encoding="utf-8-sig") as file:
                writer = csv.writer(file, delimiter=";")
                writer.writerow(HEADER)
                for i in range(40):
                    row = {name: "" for name in HEADER}
                    row.update({"время_видео_с": i * 0.7, "лицо_найдено": "да", "взгляд": "Центр",
                                "моргания": i // 3, "мимика": "Нейтрально", "микровыражение": "нет",
                                "покраснение_общее": 0.02, "напряжение_статус": "спокойно",
                                "напряжение_общее": 0.1})
                    writer.writerow([row[name] for name in HEADER])
            summary = summary_from_csv(path)
            self.assertAlmostEqual(summary["face_share"], 1.0, delta=0.01)
            self.assertEqual(summary["blink_total"], 13)
            self.assertAlmostEqual(summary["deviation_mean"], 0.1, delta=1e-6)

    def test_delete_outside_results_refused(self):
        with tempfile.TemporaryDirectory() as results, tempfile.TemporaryDirectory() as other:
            record = SessionRecord(path=other, name="x", kind="webcam", source_label="", started=None)
            with self.assertRaises(ValueError):
                delete_session(record, results)
            self.assertTrue(os.path.isdir(other))


if __name__ == "__main__":
    unittest.main()
