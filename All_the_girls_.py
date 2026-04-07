import cv2
import sounddevice as sd
import numpy as np
import threading
import queue
import textwrap
import re
from difflib import SequenceMatcher
from collections import deque

import mediapipe as mp
from faster_whisper import WhisperModel
from PIL import ImageFont, ImageDraw, Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class AudioModule:
    def __init__(self, samplerate=16000, channels=1, blocksize=1024):
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize

        self.stream = None
        self.lock = threading.Lock()
        self.volume = 0.0
        self.audio_queue = queue.Queue()

    def callback(self, indata, frames, time_info, status):
        audio = indata.copy()

        volume = float(np.sqrt(np.mean(audio ** 2)))
        with self.lock:
            self.volume = volume

        self.audio_queue.put(audio)

    def start(self):
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            blocksize=self.blocksize,
            dtype="float32",
            callback=self.callback
        )
        self.stream.start()

    def get_volume(self):
        with self.lock:
            return self.volume

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()


class SafeMeaningCorrectionModule:
    def __init__(self, model_name="cointegrated/rut5-small-normalizer"):
        print("Загрузка модели коррекции текста...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.lock = threading.Lock()
        print("Модель коррекции загружена")

    def normalize_spaces(self, text):
        return re.sub(r"\s+", " ", text.strip())

    def basic_cleanup(self, text):
        text = self.normalize_spaces(text)

        if not text:
            return text

        text = text.replace(" ,", ",")
        text = text.replace(" .", ".")
        text = text.replace(" !", "!")
        text = text.replace(" ?", "?")
        text = text.replace(" :", ":")
        text = text.replace(" ;", ";")

        if text and text[0].isalpha():
            text = text[0].upper() + text[1:]

        return text

    def looks_broken(self, text):
        if not text:
            return False

        words = text.split()
        if len(words) < 3:
            return False

        lower = text.lower()

        repeated_words = 0
        for i in range(1, len(words)):
            if words[i].lower() == words[i - 1].lower():
                repeated_words += 1

        short_words = sum(1 for w in words if len(w) <= 2)
        short_ratio = short_words / max(len(words), 1)

        repeated_chars = bool(re.search(r"(.)\1\1", lower))
        no_punctuation = not any(ch in text for ch in ".!?,")
        many_fillers = any(
            filler in lower for filler in ["ээ", "мм", "ну ну", "а а", "это это"]
        )

        return (
            repeated_words >= 1
            or short_ratio > 0.45
            or repeated_chars
            or no_punctuation
            or many_fillers
        )

    def safe_accept(self, original, corrected):
        original = self.normalize_spaces(original)
        corrected = self.normalize_spaces(corrected)

        if not corrected:
            return False

        similarity = SequenceMatcher(None, original.lower(), corrected.lower()).ratio()

        original_words = original.split()
        corrected_words = corrected.split()

        if not corrected_words:
            return False

        length_ratio = len(corrected_words) / max(len(original_words), 1)

        if similarity < 0.45:
            return False

        if length_ratio < 0.6 or length_ratio > 1.6:
            return False

        return True

    def generate_correction(self, text):
        with self.lock:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128
            )

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=4,
                do_sample=False
            )

            corrected = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            ).strip()

        return corrected

    def correct(self, text):
        text = self.basic_cleanup(text)

        if not text:
            return text

        if not self.looks_broken(text):
            return text

        try:
            corrected = self.generate_correction(text)
            corrected = self.basic_cleanup(corrected)

            if self.safe_accept(text, corrected):
                return corrected

            return text

        except Exception as error:
            print("Ошибка коррекции:", error)
            return text


class SubtitleStorage:
    def __init__(self, live_file="subtitles_live.txt", final_file="subtitles.txt"):
        self.live_file = live_file
        self.final_file = final_file
        self.lock = threading.Lock()

        with open(self.live_file, "w", encoding="utf-8") as file:
            file.write("")

        with open(self.final_file, "w", encoding="utf-8") as file:
            file.write("")

        self.last_live_line = ""
        self.last_final_line = ""

    def append_live(self, text):
        text = text.strip()
        if not text:
            return

        with self.lock:
            if text == self.last_live_line:
                return

            self.last_live_line = text
            with open(self.live_file, "a", encoding="utf-8") as file:
                file.write(text + "\n")

    def append_final(self, text):
        text = text.strip()
        if not text:
            return

        with self.lock:
            if text == self.last_final_line:
                return

            self.last_final_line = text
            with open(self.final_file, "a", encoding="utf-8") as file:
                file.write(text + "\n")


class WhisperSubtitleModule:
    def __init__(self, model_size="small"):
        print("Загрузка Whisper модели...")
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8"
        )
        print("Whisper загружен")

        self.lock = threading.Lock()
        self.current_live_text = ""
        self.final_lines = []
        self.last_display_text = ""

    def normalize_text(self, text):
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def update_live_text(self, text):
        text = self.normalize_text(text)
        if not text:
            return

        with self.lock:
            self.current_live_text = text

            if text != self.last_display_text:
                self.final_lines.append(text)
                self.final_lines = self.final_lines[-8:]
                self.last_display_text = text

    def get_display_lines(self):
        with self.lock:
            lines = self.final_lines[-2:].copy()

            if self.current_live_text:
                if not lines or lines[-1] != self.current_live_text:
                    lines.append(self.current_live_text)

            return lines[-2:]


class EyeTrackerModule:
    LEFT_EYE_EAR = [33, 159, 158, 133, 153, 145]
    RIGHT_EYE_EAR = [362, 386, 385, 263, 380, 374]

    LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_CONTOUR = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]

    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]

    LEFT_EYE_CORNERS = [33, 133]
    RIGHT_EYE_CORNERS = [362, 263]

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.ear_threshold = 0.21
        self.min_closed_frames = 2

        self.closed_frames = 0
        self.blink_count = 0
        self.blink_in_progress = False
        self.ear_history = deque(maxlen=5)

        self.last_gaze = "центр"
        self.last_ear = 0.0

        try:
            self.font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
        except Exception:
            self.font = ImageFont.load_default()

    def close(self):
        if self.face_mesh is not None:
            self.face_mesh.close()

    def to_pixel(self, landmark, width, height):
        x = min(max(int(landmark.x * width), 0), width - 1)
        y = min(max(int(landmark.y * height), 0), height - 1)
        return (x, y)

    def get_points(self, face_landmarks, indices, width, height):
        points = []
        for index in indices:
            landmark = face_landmarks.landmark[index]
            points.append(self.to_pixel(landmark, width, height))
        return points

    def distance(self, p1, p2):
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def compute_ear(self, points):
        p1, p2, p3, p4, p5, p6 = points

        horizontal = self.distance(p1, p4)
        if horizontal == 0:
            return 0.0

        vertical_1 = self.distance(p2, p6)
        vertical_2 = self.distance(p3, p5)

        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def estimate_gaze_ratio(self, iris_points, corner_points):
        iris_center = np.mean(np.array(iris_points), axis=0)
        xs = [point[0] for point in corner_points]

        min_x = float(min(xs))
        max_x = float(max(xs))
        width = max_x - min_x

        if width <= 1:
            return 0.5

        ratio = (iris_center[0] - min_x) / width
        return float(np.clip(ratio, 0.0, 1.0))

    def classify_gaze(self, ratio):
        if ratio < 0.42:
            return "влево"
        if ratio > 0.58:
            return "вправо"
        return "центр"

    def update_blink_counter(self, ear):
        self.ear_history.append(ear)
        smooth_ear = float(sum(self.ear_history) / len(self.ear_history))
        self.last_ear = smooth_ear

        if smooth_ear < self.ear_threshold:
            self.closed_frames += 1
            if self.closed_frames >= self.min_closed_frames:
                self.blink_in_progress = True
        else:
            if self.blink_in_progress:
                self.blink_count += 1
            self.closed_frames = 0
            self.blink_in_progress = False

    def draw_eye_shape(self, frame, points, color):
        if len(points) < 2:
            return
        contour = np.array(points, dtype=np.int32)
        cv2.polylines(frame, [contour], True, color, 1, cv2.LINE_AA)

    def draw_iris(self, frame, points):
        if not points:
            return

        center = np.mean(np.array(points), axis=0).astype(int)
        radius = max(2, int(np.mean([
            self.distance(center, point) for point in points
        ])))

        cv2.circle(frame, tuple(center), radius, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, tuple(center), 1, (0, 255, 255), -1, cv2.LINE_AA)

    def draw_panel(self, frame):
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        x1, y1 = 20, 50
        x2, y2 = 335, 150

        draw.rounded_rectangle(
            (x1, y1, x2, y2),
            radius=10,
            fill=(0, 0, 0),
            outline=(255, 255, 255),
            width=2
        )

        draw.text(
            (x1 + 12, y1 + 10),
            f"EAR: {self.last_ear:.3f}",
            font=self.font,
            fill=(255, 255, 255)
        )
        draw.text(
            (x1 + 12, y1 + 40),
            f"Направление: {self.last_gaze}",
            font=self.font,
            fill=(255, 255, 255)
        )
        draw.text(
            (x1 + 12, y1 + 70),
            f"Моргания: {self.blink_count}",
            font=self.font,
            fill=(255, 255, 255)
        )

        return np.array(img)

    def process(self, frame):
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            frame = self.draw_panel(frame)
            return frame

        face_landmarks = results.multi_face_landmarks[0]

        left_eye_points = self.get_points(face_landmarks, self.LEFT_EYE_EAR, width, height)
        right_eye_points = self.get_points(face_landmarks, self.RIGHT_EYE_EAR, width, height)

        left_eye_contour = self.get_points(face_landmarks, self.LEFT_EYE_CONTOUR, width, height)
        right_eye_contour = self.get_points(face_landmarks, self.RIGHT_EYE_CONTOUR, width, height)

        left_iris = self.get_points(face_landmarks, self.LEFT_IRIS, width, height)
        right_iris = self.get_points(face_landmarks, self.RIGHT_IRIS, width, height)

        left_corners = self.get_points(face_landmarks, self.LEFT_EYE_CORNERS, width, height)
        right_corners = self.get_points(face_landmarks, self.RIGHT_EYE_CORNERS, width, height)

        left_ear = self.compute_ear(left_eye_points)
        right_ear = self.compute_ear(right_eye_points)
        avg_ear = (left_ear + right_ear) / 2.0

        self.update_blink_counter(avg_ear)

        left_ratio = self.estimate_gaze_ratio(left_iris, left_corners)
        right_ratio = self.estimate_gaze_ratio(right_iris, right_corners)
        gaze_ratio = (left_ratio + right_ratio) / 2.0
        self.last_gaze = self.classify_gaze(gaze_ratio)

        self.draw_eye_shape(frame, left_eye_contour, (255, 0, 255))
        self.draw_eye_shape(frame, right_eye_contour, (255, 0, 255))
        self.draw_iris(frame, left_iris)
        self.draw_iris(frame, right_iris)

        frame = self.draw_panel(frame)
        return frame


class App:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.audio = AudioModule()
        self.whisper = WhisperSubtitleModule(model_size="small")
        self.corrector = SafeMeaningCorrectionModule()
        self.eye_tracker = EyeTrackerModule()
        self.storage = SubtitleStorage(
            live_file="subtitles_live.txt",
            final_file="subtitles.txt"
        )

        self.running = True
        self.audio_buffer = []
        self.buffer_seconds = 2.0

        try:
            self.font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
        except Exception:
            self.font = ImageFont.load_default()

    def draw_bar(self, frame, volume):
        bar_width = int(min(volume * 2500, 180))

        cv2.rectangle(frame, (20, 20), (210, 38), (255, 255, 255), 2)
        cv2.rectangle(frame, (20, 20), (20 + bar_width, 38), (0, 255, 0), -1)

        return frame

    def prepare_lines_for_display(self, lines, width=30, max_lines=2):
        wrapped = []

        for line in lines:
            wrapped.extend(textwrap.wrap(line, width=width) or [""])

        return wrapped[-max_lines:]

    def draw_subtitles(self, frame, lines):
        if not lines:
            return frame

        lines = self.prepare_lines_for_display(lines, width=30, max_lines=2)

        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        box_width = min(540, frame_width - 60)
        box_height = 64
        x1 = (frame_width - box_width) // 2
        y1 = frame_height - box_height - 16
        x2 = x1 + box_width
        y2 = y1 + box_height

        draw.rounded_rectangle(
            (x1, y1, x2, y2),
            radius=12,
            fill=(0, 0, 0)
        )

        y_text = y1 + 8
        for line in lines:
            draw.text(
                (x1 + 14, y_text),
                line,
                font=self.font,
                fill=(255, 255, 255)
            )
            y_text += 24

        return np.array(img)

    def collect_transcription(self, segments):
        text_parts = []
        avg_logprobs = []
        no_speech_probs = []

        for segment in segments:
            part = segment.text.strip()
            if part:
                text_parts.append(part)

            if hasattr(segment, "avg_logprob"):
                avg_logprobs.append(segment.avg_logprob)

            if hasattr(segment, "no_speech_prob"):
                no_speech_probs.append(segment.no_speech_prob)

        text = " ".join(text_parts).strip()

        avg_logprob = None
        if avg_logprobs:
            avg_logprob = sum(avg_logprobs) / len(avg_logprobs)

        avg_no_speech_prob = None
        if no_speech_probs:
            avg_no_speech_prob = sum(no_speech_probs) / len(no_speech_probs)

        return text, avg_logprob, avg_no_speech_prob

    def should_apply_meaning_correction(self, text, avg_logprob, avg_no_speech_prob):
        if not text:
            return False

        words = text.split()
        if len(words) < 3:
            return False

        broken = self.corrector.looks_broken(text)

        low_confidence = False
        if avg_logprob is not None and avg_logprob < -0.8:
            low_confidence = True

        suspicious_silence = False
        if avg_no_speech_prob is not None and avg_no_speech_prob > 0.5:
            suspicious_silence = True

        return broken or low_confidence or suspicious_silence

    def process_live_text(self, text):
        text = self.corrector.basic_cleanup(text)

        if not text:
            return text

        words = text.split()

        if len(words) <= 2:
            return text

        if len(words) > 14:
            text = " ".join(words[:14])

        return text

    def process_final_text(self, text, avg_logprob, avg_no_speech_prob):
        cleaned = self.corrector.basic_cleanup(text)

        if not cleaned:
            return cleaned

        if self.should_apply_meaning_correction(
            cleaned,
            avg_logprob,
            avg_no_speech_prob
        ):
            return self.corrector.correct(cleaned)

        return cleaned

    def whisper_loop(self):
        target_samples = int(self.audio.samplerate * self.buffer_seconds)

        while self.running:
            try:
                chunk = self.audio.audio_queue.get(timeout=0.1)
                self.audio_buffer.append(chunk)

                total_samples = sum(part.shape[0] for part in self.audio_buffer)

                if total_samples >= target_samples:
                    audio_data = np.concatenate(self.audio_buffer, axis=0).flatten()
                    self.audio_buffer = []

                    segments, _ = self.whisper.model.transcribe(
                        audio_data,
                        language="ru",
                        vad_filter=True,
                        beam_size=2,
                        best_of=2,
                        condition_on_previous_text=False
                    )

                    new_text, avg_logprob, avg_no_speech_prob = self.collect_transcription(segments)

                    if new_text:
                        live_text = self.process_live_text(new_text)
                        final_text = self.process_final_text(
                            new_text,
                            avg_logprob,
                            avg_no_speech_prob
                        )

                        self.whisper.update_live_text(live_text)
                        self.storage.append_live(live_text)
                        self.storage.append_final(final_text)

            except queue.Empty:
                continue
            except Exception as error:
                print("Ошибка распознавания:", error)

    def run(self):
        if not self.cap.isOpened():
            print("Не удалось открыть веб-камеру")
            return

        self.audio.start()

        whisper_thread = threading.Thread(target=self.whisper_loop, daemon=True)
        whisper_thread.start()

        print("Видео, микрофон, субтитры и отслеживание глаз запущены")
        print("Нажмите 'q' для выхода")
        print("Быстрые субтитры: subtitles_live.txt")
        print("Улучшенный текст: subtitles.txt")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Ошибка получения кадра")
                break

            volume = self.audio.get_volume()
            lines = self.whisper.get_display_lines()

            frame = self.draw_bar(frame, volume)
            frame = self.eye_tracker.process(frame)
            frame = self.draw_subtitles(frame, lines)

            cv2.imshow("Webcam", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.running = False
        self.cap.release()
        self.audio.stop()
        self.eye_tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = App()
    app.run()