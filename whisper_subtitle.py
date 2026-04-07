import re
import threading

from faster_whisper import WhisperModel


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
