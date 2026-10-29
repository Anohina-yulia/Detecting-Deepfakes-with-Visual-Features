# -*- coding: utf-8 -*-
import re
import threading
import time
from difflib import SequenceMatcher

LIVE_TTL = 5.0  # сколько секунд живой текст остаётся на экране без обновлений


class WhisperSubtitleModule:
    def __init__(self, model_size="small"):
        print("Загрузка модели Whisper...")
        from faster_whisper import WhisperModel

        self.model = self._load_model(WhisperModel, model_size)
        print("Whisper загружен.")

        self.lock = threading.Lock()
        self.current_live_text = ""
        self.live_time = 0.0
        self.final_lines = []

        self.last_live_text = ""
        self.last_final_text = ""  # последнее ОКНО распознавания (не накопленный текст)
        self.last_raw_text = ""

    @staticmethod
    def _load_model(WhisperModel, model_size):
        try:
            model = WhisperModel(model_size, device="cpu", compute_type="int8",
                                 local_files_only=True)
            print("Модель Whisper найдена в локальном кэше.")
            return model
        except Exception:
            pass

        print("Локального кэша нет, скачиваю модель Whisper (один раз)...")
        return WhisperModel(model_size, device="cpu", compute_type="int8")

    # ---------- очистка ----------

    @staticmethod
    def normalize_text(text):
        text = re.sub(r"\s+", " ", (text or "").strip())
        for mark in (",", ".", "!", "?", ":", ";"):
            text = text.replace(f" {mark}", mark)
        return text.strip()

    def cleanup_text(self, text):
        """Очистка для показа на экране. Для анализа речи используется сырой текст."""
        text = self.normalize_text(text)
        if not text:
            return ""
        if text.lower() in {"ээ", "эм", "м", "а", "ну", "угу", "ага"}:
            return ""

        cleaned = []
        for word in text.split():
            if cleaned and word.lower() == cleaned[-1].lower():
                continue
            cleaned.append(word)

        text = self.normalize_text(" ".join(cleaned))
        if text and text[0].isalpha():
            text = text[0].upper() + text[1:]
        return text

    def is_too_similar(self, a, b, threshold=0.88):
        a, b = self.normalize_text(a).lower(), self.normalize_text(b).lower()
        if not a or not b:
            return False
        return SequenceMatcher(None, a, b).ratio() >= threshold

    # ---------- выделение нового хвоста ----------

    @staticmethod
    def _key(word):
        return re.sub(r"[^\w]", "", word.lower())

    def new_tail(self, previous, current):
        """Часть current, которой не было в previous (сравнение по словам)."""
        if not previous:
            return current

        cur_words = current.split()
        cur_keys = [self._key(w) for w in cur_words]
        prev_keys = [self._key(w) for w in previous.split()]
        n = len(cur_keys)
        if n == 0:
            return ""

        # текущее окно целиком уже внутри предыдущего
        for i in range(len(prev_keys) - n + 1):
            if prev_keys[i:i + n] == cur_keys:
                return ""

        # конец предыдущего совпадает с началом текущего
        for k in range(min(len(prev_keys), n), 0, -1):
            if prev_keys[-k:] == cur_keys[:k] and (k >= 2 or n == 1):
                return " ".join(cur_words[k:])

        return current

    # ---------- живой и финальный текст ----------

    def update_live_text(self, text):
        text = self.cleanup_text(text)
        if not text:
            return ""

        with self.lock:
            self.live_time = time.time()
            if self.last_live_text and self.is_too_similar(self.last_live_text, text, 0.93):
                return self.current_live_text
            self.current_live_text = text
            self.last_live_text = text
            return text

    def commit_final_text(self, text, raw_text=None):
        """Возвращает (новый_хвост_для_показа, новый_хвост_сырого_текста) или None."""
        text = self.cleanup_text(text)
        if not text:
            return None
        if len(text.split()) < 2 and not any(ch in text for ch in ".!?"):
            return None

        raw = self.normalize_text(raw_text or text)

        with self.lock:
            if self.last_final_text and self.is_too_similar(self.last_final_text, text, 0.90):
                return None

            tail = self.normalize_text(self.new_tail(self.last_final_text, text))
            raw_tail = self.normalize_text(self.new_tail(self.last_raw_text, raw))
            self.last_final_text = text
            self.last_raw_text = raw

            if not tail:
                return None
            if tail[0].isalpha():
                tail = tail[0].upper() + tail[1:]

            self.final_lines = (self.final_lines + [tail])[-8:]
            self.current_live_text = text
            self.last_live_text = text
            self.live_time = time.time()
            return tail, raw_tail

    def get_display_lines(self):
        with self.lock:
            if self.current_live_text and (time.time() - self.live_time) <= LIVE_TTL:
                return [self.current_live_text]
            return []