import cv2
import sounddevice as sd
import numpy as np
import threading
import queue
import textwrap
from faster_whisper import WhisperModel
from PIL import ImageFont, ImageDraw, Image


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


class WhisperSubtitleModule:
    def __init__(self, model_size="small", subtitles_file="subtitles.txt"):
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8"
        )

        self.subtitles_file = subtitles_file
        self.lock = threading.Lock()

        self.current_live_text = ""
        self.final_lines = []
        self.last_saved_text = ""

        with open(self.subtitles_file, "w", encoding="utf-8") as file:
            file.write("")

    def normalize_text(self, text):
        text = text.strip()
        while "  " in text:
            text = text.replace("  ", " ")
        return text

    def update_text(self, text):
        text = self.normalize_text(text)
        if not text:
            return

        with self.lock:
            self.current_live_text = text

            if text != self.last_saved_text:
                self.final_lines.append(text)
                self.final_lines = self.final_lines[-8:]
                self.last_saved_text = text

                with open(self.subtitles_file, "a", encoding="utf-8") as file:
                    file.write(text + "\n")

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

        self.running = True
        self.audio_buffer = []
        self.buffer_seconds = 2.0

        try:
            self.font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
        except Exception:
            self.font = ImageFont.load_default()

    def draw_bar(self, frame, volume):
        bar_width = int(min(volume * 2500, 220))

        cv2.rectangle(frame, (20, 20), (250, 40), (255, 255, 255), 2)
        cv2.rectangle(frame, (20, 20), (20 + bar_width, 40), (0, 255, 0), -1)

        return frame

    def prepare_lines_for_display(self, lines, width=34, max_lines=2):
        wrapped = []

        for line in lines:
            wrapped.extend(textwrap.wrap(line, width=width) or [""])

        return wrapped[-max_lines:]

    def draw_subtitles(self, frame, lines):
        if not lines:
            return frame

        lines = self.prepare_lines_for_display(lines, width=34, max_lines=2)

        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        box_width = min(700, frame_width - 40)
        box_height = 80
        x1 = (frame_width - box_width) // 2
        y1 = frame_height - box_height - 20
        x2 = x1 + box_width
        y2 = y1 + box_height

        draw.rectangle((x1, y1, x2, y2), fill=(0, 0, 0))

        y_text = y1 + 8
        for line in lines:
            draw.text((x1 + 15, y_text), line, font=self.font, fill=(255, 255, 255))
            y_text += 28

        return np.array(img)

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
                        condition_on_previous_text=True
                    )

                    text_parts = []
                    for segment in segments:
                        part = segment.text.strip()
                        if part:
                            text_parts.append(part)

                    new_text = " ".join(text_parts).strip()

                    if new_text:
                        self.whisper.update_text(new_text)

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