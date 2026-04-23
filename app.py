# -*- coding: utf-8 -*-
import queue
import textwrap
import threading

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from audio_module import AudioModule
from text_correction import SafeMeaningCorrectionModule
from subtitle_storage import SubtitleStorage
from whisper_subtitle import WhisperSubtitleModule
from neural_face_analyzer import NeuralFaceAnalyzerModule
from eye_tracker import EyeTrackerModule
from mimic_analyzer import MimicAnalyzerModule
from micro_expression import MicroExpressionModule
from analysis_logger import AnalysisLogger
from redness_analyzer import RednessAnalyzerModule
from suspicion_analyzer import SuspicionAnalyzerModule
from quality_analyzer import QualityAnalyzerModule
from video_analysis_worker import VideoAnalysisWorker
from models import MimicData, RednessData


class App:
    def __init__(self):
        cv2.setUseOptimized(True)

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.audio = AudioModule()
        self.whisper = WhisperSubtitleModule(model_size="small")
        self.corrector = SafeMeaningCorrectionModule()
        self.storage = SubtitleStorage(
            live_file="subtitles_live.txt",
            final_file="subtitles.txt"
        )

        # Одна модель для UI: лицо/глаза/рамка каждый кадр
        self.neural_face_ui = NeuralFaceAnalyzerModule(
            model_path="models/face_landmarker.task"
        )

        # Вторая модель для фонового анализа: мимика/покраснения
        self.neural_face_bg = NeuralFaceAnalyzerModule(
            model_path="models/face_landmarker.task"
        )

        self.eye_tracker = EyeTrackerModule()
        self.mimic_analyzer = MimicAnalyzerModule()
        self.micro_module = MicroExpressionModule()
        self.redness_analyzer = RednessAnalyzerModule()
        self.suspicion_analyzer = SuspicionAnalyzerModule()
        self.quality_analyzer = QualityAnalyzerModule()

        self.analysis_worker = VideoAnalysisWorker(
            neural_face=self.neural_face_bg,
            mimic_analyzer=self.mimic_analyzer,
            micro_module=self.micro_module,
            redness_analyzer=self.redness_analyzer,
            quality_analyzer=self.quality_analyzer,
            submit_every_n=2
        )

        self.analysis_logger = AnalysisLogger(
            filepath="analysis_log.csv",
            min_interval_sec=0.7
        )

        self.running = True
        self.audio_buffer = []
        self.buffer_seconds = 2.2
        self.frame_index = 0

        self.last_mimic_data = MimicData()
        self.last_micro_event = None
        self.last_redness_data = RednessData(status="ожидание")
        self.last_redness_boxes = {}
        self.last_suspicion_data = None

        try:
            self.font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
        except Exception:
            self.font = ImageFont.load_default()

    def draw_bar(self, frame, volume):
        bar_width = int(min(volume * 2500, 180))
        cv2.rectangle(frame, (20, 20), (210, 38), (255, 255, 255), 2)
        cv2.rectangle(frame, (20, 20), (20 + bar_width, 38), (0, 255, 0), -1)
        return frame

    def draw_face_box(self, frame, face_data):
        if not face_data.detected or not face_data.bbox:
            return frame

        x1, y1, x2, y2 = face_data.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
        return frame

    def prepare_lines_for_display(self, lines, width=34, max_lines=2):
        wrapped = []
        for line in lines:
            wrapped.extend(textwrap.wrap(line, width=width) or [""])
        return wrapped[-max_lines:]

    def draw_subtitles(self, frame, lines, face_data=None):
        if not lines:
            return frame

        lines = self.prepare_lines_for_display(lines, width=34, max_lines=2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)

        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        box_width = min(560, frame_width - 80)
        box_height = 58 if len(lines) == 1 else 82

        x1 = (frame_width - box_width) // 2
        y1 = frame_height - box_height - 12
        x2 = x1 + box_width
        y2 = y1 + box_height

        if face_data is not None and face_data.detected and face_data.bbox:
            fx1, fy1, fx2, fy2 = face_data.bbox

            overlap_x = not (fx2 < x1 or fx1 > x2)
            overlap_y = not (fy2 < y1 or fy1 > y2)

            if overlap_x and overlap_y:
                y1 = max(fy1 - box_height - 10, 10)
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

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

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

        if len(text.split()) < 3:
            return False

        broken = self.corrector.looks_broken(text)
        low_confidence = avg_logprob is not None and avg_logprob < -0.8
        suspicious_silence = avg_no_speech_prob is not None and avg_no_speech_prob > 0.5

        return broken or low_confidence or suspicious_silence

    def process_live_text(self, text):
        text = self.corrector.basic_cleanup(text)

        if not text:
            return text

        words = text.split()
        if len(words) > 14:
            text = " ".join(words[:14])

        return text

    def process_final_text(self, text, avg_logprob, avg_no_speech_prob):
        cleaned = self.corrector.basic_cleanup(text)

        if not cleaned:
            return cleaned

        if self.should_apply_meaning_correction(cleaned, avg_logprob, avg_no_speech_prob):
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
                        beam_size=1,
                        best_of=1,
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

        print("Видео, микрофон, субтитры, лицо, взгляд, мимика, покраснения и индикатор запущены")
        print("Калибровка идёт автоматически после запуска")
        print("Нажмите 'q' для выхода")
        print("Нажмите 'c' для повторной калибровки мимики")
        print("Нажмите 'r' для повторной калибровки покраснений")
        print("Быстрые субтитры: subtitles_live.txt")
        print("Улучшенный текст: subtitles.txt")
        print("Лог анализа: analysis_log.csv")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Ошибка получения кадра")
                break

            volume = self.audio.get_volume()
            lines = self.whisper.get_display_lines()
            subtitle_text = lines[-1] if lines else ""

            # СВЕЖЕЕ лицо для глаз/рамки каждый кадр
            current_face_data, _ = self.neural_face_ui.process(frame, self.frame_index)

            # Фоновый тяжёлый анализ
            self.analysis_worker.submit(frame, self.frame_index)
            snapshot = self.analysis_worker.get_latest()

            if snapshot.mimic_data is not None:
                self.last_mimic_data = snapshot.mimic_data
            if snapshot.redness_data is not None:
                self.last_redness_data = snapshot.redness_data
            if snapshot.redness_boxes is not None:
                self.last_redness_boxes = snapshot.redness_boxes
            if snapshot.micro_event is not None or self.last_micro_event is None:
                self.last_micro_event = snapshot.micro_event

            neural_mimic_data = self.last_mimic_data
            redness_data = self.last_redness_data
            micro_event = self.last_micro_event

            frame = self.draw_bar(frame, volume)
            frame = self.draw_face_box(frame, current_face_data)
            frame = self.eye_tracker.process(frame, current_face_data)

            suspicion_data = self.suspicion_analyzer.process(
                eye_tracker=self.eye_tracker,
                mimic_data=neural_mimic_data,
                subtitle_text=subtitle_text
            )
            self.last_suspicion_data = suspicion_data

            self.analysis_logger.log(
                face_data=current_face_data,
                eye_tracker=self.eye_tracker,
                mimic_data=neural_mimic_data,
                micro_event=micro_event,
                subtitle_text=subtitle_text,
                redness_data=redness_data,
                suspicion_data=suspicion_data
            )

            frame = self.mimic_analyzer.draw_panel(frame, neural_mimic_data, micro_event)
            frame = self.redness_analyzer.draw_panel(frame, redness_data)
            frame = self.suspicion_analyzer.draw_panel(frame, suspicion_data)
            frame = self.draw_subtitles(frame, lines, current_face_data)

            cv2.imshow("Webcam", frame)

            self.frame_index += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                self.mimic_analyzer.start_calibration()
            elif key == ord("r"):
                self.redness_analyzer.start_calibration()

        self.running = False
        self.analysis_worker.close()
        self.cap.release()
        self.audio.stop()
        self.neural_face_ui.close()
        self.neural_face_bg.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = App()
    app.run()