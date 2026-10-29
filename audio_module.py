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

    def start(self):
        import sounddevice as sd

        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            blocksize=self.blocksize,
            dtype="float32",
            callback=self.callback,
        )
        self.stream.start()

    def get_volume(self):
        with self.lock:
            return self.volume

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None