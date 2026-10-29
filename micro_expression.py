# -*- coding: utf-8 -*-
from collections import deque

from models import MicroEvent

# (название события, поле MimicData, порог значения)
CHECKS = [
    ("быстрый подъём бровей", "brow_raise", 0.45),
    ("быстрое нахмуривание", "brow_frown", 0.40),
    ("краткая улыбка", "smile", 0.45),
    ("резкое открытие рта", "mouth_open", 0.45),
    ("резкий прищур", "eye_squint", 0.45),
]


class MicroExpressionModule:
    """Ищет резкие скачки мимики.

    * Скорость считается между текущим кадром и кадром, который старше не менее
      min_dt секунд (так шум двух соседних кадров не даёт ложных срабатываний).
    * Время берётся из таймлайна кадров, поэтому для видеофайла всё корректно.
    * Событие показывается event_ttl секунд, а не бесконечно.
    """

    def __init__(self, maxlen=40, cooldown_seconds=0.8, min_dt=0.08,
                 max_dt=0.4, speed_threshold=2.5, event_ttl=2.0):
        self.history = deque(maxlen=maxlen)
        self.events = deque(maxlen=5)
        self.cooldown_seconds = cooldown_seconds
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.speed_threshold = speed_threshold
        self.event_ttl = event_ttl
        self.last_event_time = -1e9

    def update(self, mimic_data, timestamp):
        self.history.append((timestamp, mimic_data))

        if (timestamp - self.last_event_time) < self.cooldown_seconds:
            return self.get_active(timestamp)

        reference = None
        for t, m in reversed(self.history):
            if timestamp - t >= self.min_dt:
                reference = (t, m)
                break

        if reference is None:
            return self.get_active(timestamp)

        t_ref, prev = reference
        dt = timestamp - t_ref
        if dt > self.max_dt:
            return self.get_active(timestamp)

        for name, attr, threshold in CHECKS:
            current = getattr(mimic_data, attr)
            speed = (current - getattr(prev, attr)) / dt
            if speed > self.speed_threshold and current > threshold:
                event = MicroEvent(name=name, strength=current,
                                   duration_ms=dt * 1000.0, timestamp=timestamp)
                self.events.append(event)
                self.last_event_time = timestamp
                return event

        return self.get_active(timestamp)

    def get_active(self, now):
        """Последнее событие, если оно произошло не позже event_ttl секунд назад."""
        if not self.events:
            return None
        event = self.events[-1]
        return event if (now - event.timestamp) <= self.event_ttl else None