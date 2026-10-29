# -*- coding: utf-8 -*-
import queue
import threading
import time

import numpy as np


class AudioModule:
    """Источник звука: микрофон (start) или внешняя подача кусков (feed).

    Очередь ограничена: если распознавание не успевает, старые блоки
    выбрасываются, а не копятся в памяти.
    """

    def __init__(self, samplerate=16000, channels=1, blocksize=1024, max_queue=400):
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize

        self.stream = None
        self.lock = threading.Lock()
        self.volume = 0.0
        self.audio_queue = queue.Queue(maxsize=max_queue)
        self._last_status_print = 0.0

    def feed(self, chunk):
        audio = np.array(chunk, dtype=np.float32, copy=True).reshape(-1, 1)
        volume = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
        with self.lock:
            self.volume = volume

        try:
            self.audio_queue.put_nowait(audio)
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(audio)
            except (queue.Empty, queue.Full):
                pass

    def callback(self, indata, frames, time_info, status):
        if status and time.time() - self._last_status_print > 5.0:
            print("Аудио:", status)
            self._last_status_print = time.time()
        self.feed(indata)

    def start(self, device=None):
        """device - индекс устройства ввода sounddevice (None - по умолчанию)."""
        import sounddevice as sd

        if self.stream is not None:
            return
        stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            blocksize=self.blocksize,
            dtype="float32",
            device=device,
            callback=self.callback,
        )
        stream.start()
        self.stream = stream

    @property
    def active(self):
        return self.stream is not None

    def get_volume(self):
        with self.lock:
            return self.volume

    def clear_queue(self):
        """Выбросить накопленные блоки (после паузы старый звук не нужен)."""
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def stop(self):
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as error:
                print("Ошибка при закрытии микрофона:", error)
        with self.lock:
            self.volume = 0.0


def list_input_devices():
    """[(индекс, название)] устройств ввода; пустой список, если звук недоступен."""
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        default_api = sd.default.hostapi if hasattr(sd.default, "hostapi") else None
        result = []
        for index, info in enumerate(devices):
            if info.get("max_input_channels", 0) <= 0:
                continue
            # MME дублирует устройства других API - показываем только API по умолчанию
            if default_api is not None and default_api >= 0 and info.get("hostapi") != default_api:
                continue
            result.append((index, info.get("name", f"Устройство {index}")))
        return result
    except Exception as error:
        print("Не удалось получить список микрофонов:", error)
        return []