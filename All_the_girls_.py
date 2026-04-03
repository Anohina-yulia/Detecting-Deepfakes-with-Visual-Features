import cv2
import sounddevice as sd
import numpy as np
import threading
import queue
import textwrap
import re
from difflib import SequenceMatcher

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


class App:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.audio = AudioModule()
        self.whisper = WhisperSubtitleModule(model_size="small")
        self.corrector = SafeMeaningCorrectionModule()
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

        print("Видео, микрофон и субтитры запущены")
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
            frame = self.draw_subtitles(frame, lines)

            cv2.imshow("Webcam", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.running = False
        self.cap.release()
        self.audio.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = App()
    app.run()