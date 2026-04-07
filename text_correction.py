import re
import threading
from difflib import SequenceMatcher

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


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
