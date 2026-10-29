# -*- coding: utf-8 -*-
"""Запуск настольного интерфейса."""
import os
import sys
import threading
import traceback

# тише логи нейросетевых библиотек и без телеметрии Hugging Face (до их импорта)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")

from PySide6.QtCore import QObject, Qt, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app_paths import LOG_DIR  # noqa: E402
from settings_store import SettingsStore  # noqa: E402
from speech_engine import SpeechModels  # noqa: E402

_bridge = None
_window = None


class LogBus(QObject):
    line = Signal(str)


class ModelsBridge(QObject):
    changed = Signal()


class TeeStream:
    """Пишет вывод модулей (print) в консоль, в logs/app.log и в журнал окна."""

    def __init__(self, original, bus, log_file):
        self.original = original
        self.bus = bus
        self.log_file = log_file
        self.encoding = "utf-8"
        self._buffer = ""
        self._lock = threading.Lock()

    def write(self, text):
        if not text:
            return 0
        if self.original is not None:
            try:
                self.original.write(text)
            except Exception:
                pass
        with self._lock:
            if self.log_file is not None:
                try:
                    self.log_file.write(text)
                    self.log_file.flush()
                except Exception:
                    pass
            self._buffer += text
            lines = self._buffer.split("\n")
            self._buffer = lines.pop()
        for line in lines:
            line = line.rstrip("\r")
            if line.strip():
                try:
                    self.bus.line.emit(line)
                except RuntimeError:
                    pass
        return len(text)

    def flush(self):
        if self.original is not None:
            try:
                self.original.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def start_model_loading(models, settings):
    if not settings["speech_enabled"]:
        models.disable()
        return

    def done():
        if _bridge is not None:
            _bridge.changed.emit()

    if models.load_async(model_size=settings["whisper_model"] or "small",
                         use_correction=bool(settings["text_correction"]),
                         allow_download=bool(settings["allow_model_download"]),
                         on_done=done):
        done()  # состояние "загрузка" - обновить интерфейс сразу


def _set_app_id():
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("AllTheGirls.Observer")
    except Exception:
        pass


def run_gui(argv=None):
    global _bridge, _window
    argv = list(sys.argv if argv is None else argv)
    _set_app_id()
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(argv)
    app.setApplicationName("All the girls")
    app.setOrganizationName("All the girls")

    from gui.icons import app_icon
    from gui.main_window import MainWindow, initial_geometry
    from gui.theme import apply_theme

    apply_theme(app)
    app.setWindowIcon(app_icon())

    bus = LogBus()
    log_file = None
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        log_file = open(os.path.join(LOG_DIR, "app.log"), "w", encoding="utf-8")
    except Exception:
        log_file = None
    sys.stdout = TeeStream(sys.stdout, bus, log_file)
    sys.stderr = TeeStream(sys.stderr, bus, log_file)

    def excepthook(exc_type, exc, tb):
        print("".join(traceback.format_exception(exc_type, exc, tb)))
        if _window is not None:
            try:
                _window.toast.show_message(f"Внутренняя ошибка: {exc}", "error", 6000)
            except Exception:
                pass

    def thread_excepthook(args):
        print(f"Ошибка в потоке {args.thread.name if args.thread else ''}:")
        print("".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)))

    sys.excepthook = excepthook
    threading.excepthook = thread_excepthook

    settings = SettingsStore()
    models = SpeechModels()
    _bridge = ModelsBridge()

    window = MainWindow(settings, models, bus, _bridge)
    _window = window
    initial_geometry(window)
    window.show()
    print("All the girls запущено. Анализ выполняется локально.")
    start_model_loading(models, settings)
    code = app.exec()
    _window = None
    if log_file is not None:
        try:
            log_file.flush()
        except Exception:
            pass
    return code
