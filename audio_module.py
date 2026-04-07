import queue
import threading

import numpy as np
import sounddevice as sd


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
