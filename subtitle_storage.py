# -*- coding: utf-8 -*-
import threading


class SubtitleStorage:
    """Запись субтитров в два файла. Каждая строка получает метку [мм:сс]
    по времени видео, чтобы сопоставлять её с analysis_log.csv."""

    def __init__(self, live_file="subtitles_live.txt", final_file="subtitles.txt",
                 time_fn=None, enabled=True):
        """enabled=False - ничего не записывать на диск (сохранение отключено)."""
        self.live_file = live_file
        self.final_file = final_file
        self.time_fn = time_fn
        self.enabled = enabled
        self.lock = threading.Lock()

        if self.enabled:
            for path in (self.live_file, self.final_file):
                with open(path, "w", encoding="utf-8") as file:
                    file.write("")

        self.last_live_line = ""
        self.last_final_line = ""

    def _prefix(self, t=None):
        """t - время произнесения фразы; без него берётся текущее время."""
        if t is None:
            if self.time_fn is None:
                return ""
            t = self.time_fn()
        total = int(t)
        return f"[{total // 60:02d}:{total % 60:02d}] "

    def append_live(self, text):
        text = (text or "").strip()
        if not text:
            return
        with self.lock:
            if text == self.last_live_line:
                return
            self.last_live_line = text
            if not self.enabled:
                return
            with open(self.live_file, "a", encoding="utf-8") as file:
                file.write(self._prefix() + text + "\n")

    def append_final(self, text, t=None):
        text = (text or "").strip()
        if not text:
            return
        with self.lock:
            if text == self.last_final_line:
                return
            self.last_final_line = text
            if not self.enabled:
                return
            with open(self.final_file, "a", encoding="utf-8") as file:
                file.write(self._prefix(t) + text + "\n")