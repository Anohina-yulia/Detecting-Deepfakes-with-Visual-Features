from collections import deque

from models import MicroEvent


class MicroExpressionModule:
    def __init__(self, maxlen=20, cooldown_seconds=0.8):
        self.history = deque(maxlen=maxlen)
        self.events = deque(maxlen=5)
        self.cooldown_seconds = cooldown_seconds
        self.last_event_time = 0.0

    def update(self, mimic_data, timestamp):
        self.history.append((timestamp, mimic_data))

        if len(self.history) < 4:
            return None

        _, prev = self.history[-2]
        t_now, cur = self.history[-1]

        dt = max(t_now - self.history[-2][0], 1e-6)

        smile_speed = (cur.smile - prev.smile) / dt
        brow_raise_speed = (cur.brow_raise - prev.brow_raise) / dt
        frown_speed = (cur.brow_frown - prev.brow_frown) / dt
        mouth_open_speed = (cur.mouth_open - prev.mouth_open) / dt
        squint_speed = (cur.eye_squint - prev.eye_squint) / dt

        if (t_now - self.last_event_time) < self.cooldown_seconds:
            return self.get_last_event()

        candidate = None

        if brow_raise_speed > 2.5 and cur.brow_raise > 0.45:
            candidate = MicroEvent(
                name="быстрый подъём бровей",
                strength=cur.brow_raise,
                duration_ms=150.0,
                timestamp=t_now
            )
        elif frown_speed > 2.5 and cur.brow_frown > 0.40:
            candidate = MicroEvent(
                name="быстрое нахмуривание",
                strength=cur.brow_frown,
                duration_ms=150.0,
                timestamp=t_now
            )
        elif smile_speed > 2.5 and cur.smile > 0.45:
            candidate = MicroEvent(
                name="краткая улыбка",
                strength=cur.smile,
                duration_ms=150.0,
                timestamp=t_now
            )
        elif mouth_open_speed > 2.5 and cur.mouth_open > 0.45:
            candidate = MicroEvent(
                name="резкое открытие рта",
                strength=cur.mouth_open,
                duration_ms=150.0,
                timestamp=t_now
            )
        elif squint_speed > 2.5 and cur.eye_squint > 0.45:
            candidate = MicroEvent(
                name="резкий прищур",
                strength=cur.eye_squint,
                duration_ms=150.0,
                timestamp=t_now
            )

        if candidate is not None:
            self.events.append(candidate)
            self.last_event_time = t_now
            return candidate

        return self.get_last_event()

    def get_last_event(self):
        if not self.events:
            return None
        return self.events[-1]