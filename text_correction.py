# -*- coding: utf-8 -*-
import re
import threading
from difflib import SequenceMatcher

FILLER_PATTERNS = [re.compile(p) for p in (
    r"\bээ+\b", r"\bэм+\b", r"\bну\b", r"\bкак бы\b", r"\bэто самое\b", r"\bв общем\b")]

# если модель переписала фразу слишком сильно - оставляем исходную
MIN_SIMILARITY = 0.6


class SafeMeaningCorrectionModule:
    def __init__(self, model_name="cointegrated/rut5-small-normalizer"):
        print("Загрузка модели коррекции текста...")
        self.lock = threading.Lock()
        self.tokenizer = None
        self.model = None

        try:
            self.tokenizer, self.model = self._load_model(model_name)
            print("Модель коррекции текста загружена.")
        except Exception as error:
            print("Модель коррекции недоступна, работаю только с базовой очисткой:", error)

    @staticmethod
    def _load_model(model_name):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
            print("Модель коррекции найдена в локальном кэше.")
            return tokenizer, model
        except Exception:
            pass

        print("Локального кэша нет, скачиваю модель коррекции...")
        return (AutoTokenizer.from_pretrained(model_name),
                AutoModelForSeq2SeqLM.from_pretrained(model_name))

    @staticmethod
    def basic_cleanup(text):
        text = re.sub(r"\s+", " ", (text or "").strip())
        for mark in (",", ".", "!", "?", ":", ";"):
            text = text.replace(f" {mark}", mark)

        result = []
        for word in text.split():
            if result and word.lower() == result[-1].lower():
                continue
            result.append(word)
        text = " ".join(result)

        if text and text[0].isalpha():
            text = text[0].upper() + text[1:]
        return text.strip()

    def looks_broken(self, text):
        text = self.basic_cleanup(text)
        words = text.split()
        if len(words) < 3:
            return False

        repeated = sum(1 for i in range(1, len(words))
                       if words[i].lower() == words[i - 1].lower())
        if repeated >= 2:
            return True

        low = text.lower()
        return any(len(p.findall(low)) >= 2 for p in FILLER_PATTERNS)

    def correct(self, text):
        text = self.basic_cleanup(text)
        if not text or self.model is None:
            return text

        try:
            with self.lock:
                inputs = self.tokenizer(text, return_tensors="pt",
                                        truncation=True, max_length=256)
                outputs = self.model.generate(**inputs, max_length=256,
                                              num_beams=2, do_sample=False)
                corrected = self.basic_cleanup(
                    self.tokenizer.decode(outputs[0], skip_special_tokens=True))

            if not corrected:
                return text

            similarity = SequenceMatcher(None, text.lower(), corrected.lower()).ratio()
            return corrected if similarity >= MIN_SIMILARITY else text

        except Exception as error:
            print("Ошибка коррекции текста:", error)
            return text