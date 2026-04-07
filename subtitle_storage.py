import threading


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
