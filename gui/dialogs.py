# -*- coding: utf-8 -*-
"""Диалоги: настройки, история сессий, журнал анализа."""
import html
import os
import threading

from PySide6.QtCore import QObject, QRectF, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFontMetrics, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import (QComboBox, QDialog, QFrame, QGridLayout, QHBoxLayout, QLineEdit,
                               QListWidget,
                               QListWidgetItem, QMessageBox, QPlainTextEdit, QScrollArea, QTabWidget,
                               QTextBrowser, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from gui.theme import C, Fonts, font, numeric_font, qc
from gui.transcript import fmt_time
from gui.widgets import Card, FieldRow, ToggleSwitch, label, make_button
from models import EV_MICRO, EV_SPEECH
from session_history import (delete_session, list_sessions, load_summary, read_transcript)

WHISPER_SIZES = [("tiny", "tiny — самая быстрая"),
                 ("base", "base — быстрая"),
                 ("small", "small — баланс"),
                 ("medium", "medium — точнее, медленнее")]


def open_path(path):
    if path and os.path.exists(path):
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))


def _hf_cache_dirs():
    dirs = []
    for env in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(env):
            dirs.append(os.environ[env])
    if os.environ.get("HF_HOME"):
        dirs.append(os.path.join(os.environ["HF_HOME"], "hub"))
    dirs.append(os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"))
    return dirs


def whisper_cached(size):
    name = f"models--Systran--faster-whisper-{size}"
    return any(os.path.isdir(os.path.join(d, name)) for d in _hf_cache_dirs())


def _section(title, subtitle=None):
    card = Card(margins=(20, 16, 20, 10))
    card.body.setSpacing(0)
    card.body.addWidget(label(title, "SectionTitle"))
    if subtitle:
        card.body.addSpacing(2)
        card.body.addWidget(label(subtitle, "Hint", wrap=True))
    card.body.addSpacing(6)
    return card


def _divider():
    line = QFrame()
    line.setFixedHeight(1)
    line.setStyleSheet(f"background: {C.BORDER};")
    return line


class _DialogBase(QDialog):
    def __init__(self, title, subtitle, parent=None, size=(720, 720)):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(*size)
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(28, 24, 28, 20)
        self.root.setSpacing(14)
        head = QVBoxLayout()
        head.setSpacing(4)
        head.addWidget(label(title, "DialogTitle"))
        if subtitle:
            head.addWidget(label(subtitle, "Muted", wrap=True))
        self.root.addLayout(head)


# ---------------------------------------------------------------- настройки

class _CameraProbe(QObject):
    done = Signal(str)

    def probe(self, index):
        def run():
            try:
                import cv2

                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
                ok = cap.isOpened() and cap.read()[0]
                cap.release()
                self.done.emit("Камера отвечает" if ok else "Камера не найдена или занята")
            except Exception as error:
                self.done.emit(f"Ошибка проверки: {error}")

        threading.Thread(target=run, daemon=True).start()


class SettingsDialog(_DialogBase):
    def __init__(self, settings, parent=None, session_active=False, results_dir=""):
        super().__init__("Настройки", "Параметры источника, распознавания речи, отображения и хранения "
                         "данных. Всё хранится локально в settings.json.", parent, (760, 780))
        self.settings = settings
        self.results_dir = results_dir

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setContentsMargins(2, 2, 10, 2)
        lay.setSpacing(14)
        scroll.setWidget(host)
        self.root.addWidget(scroll, 1)

        s = settings
        self.controls = {}

        # --- источник ---
        sec = _section("Источник видео")
        self.camera = QComboBox()
        for i in range(5):
            self.camera.addItem(f"Камера {i}", i)
        self.camera.setCurrentIndex(max(0, min(int(s["camera_index"] or 0), 4)))
        self.camera.setFixedWidth(150)
        probe_row = QWidget()
        probe_lay = QHBoxLayout(probe_row)
        probe_lay.setContentsMargins(0, 0, 0, 0)
        probe_lay.setSpacing(8)
        probe_lay.addWidget(self.camera)
        probe_button = make_button("Проверить", None, None, "sm")
        probe_lay.addWidget(probe_button)
        sec.body.addWidget(FieldRow("Веб-камера", "Номер камеры в системе. Встроенная камера обычно — 0.",
                                    probe_row))
        self.probe_label = label("", "FieldHint")
        sec.body.addWidget(self.probe_label)
        self._probe = _CameraProbe()
        self._probe.done.connect(self.probe_label.setText)
        probe_button.clicked.connect(self._run_probe)
        sec.body.addWidget(_divider())
        self._toggle(sec, "pace_video_file", "Видеофайл в реальном времени",
                     "Воспроизводить файл со скоростью записи, чтобы субтитры успевали за звуком. "
                     "Если выключить — анализ идёт максимально быстро, субтитры отстанут.")
        lay.addWidget(sec)

        # --- речь ---
        sec = _section("Речь и субтитры", "Распознавание выполняется локально моделью faster-whisper.")
        self._toggle(sec, "speech_enabled", "Распознавание речи",
                     "Микрофон включается только во время анализа и выключается на паузе.")
        sec.body.addWidget(_divider())
        self.device = QComboBox()
        self.device.setFixedWidth(260)
        self.device.addItem("По умолчанию", None)
        from audio_module import list_input_devices

        for index, name in list_input_devices():
            self.device.addItem(name, index)
        current = s["audio_device"]
        pos = self.device.findData(current)
        self.device.setCurrentIndex(pos if pos >= 0 else 0)
        sec.body.addWidget(FieldRow("Устройство ввода",
                                    "Для видеозвонка с экрана можно выбрать «Стерео микшер», "
                                    "если он включён в Windows.", self.device))
        sec.body.addWidget(_divider())
        self.model = QComboBox()
        self.model.setFixedWidth(300)
        for size, text in WHISPER_SIZES:
            note = " · скачана" if whisper_cached(size) else " · нужна загрузка"
            self.model.addItem(text + note, size)
        pos = self.model.findData(s["whisper_model"])
        self.model.setCurrentIndex(pos if pos >= 0 else 2)
        sec.body.addWidget(FieldRow("Модель Whisper", "Больше модель — точнее текст и выше нагрузка "
                                                      "на процессор.", self.model))
        sec.body.addWidget(_divider())
        self._toggle(sec, "text_correction", "Коррекция текста нейросетью",
                     "Модель rut5 исправляет неразборчивые фразы. Требует ~1 ГБ памяти; "
                     "без неё остаётся базовая очистка.")
        sec.body.addWidget(_divider())
        self._toggle(sec, "allow_model_download", "Разрешить загрузку моделей",
                     "Только если нужной модели нет на компьютере. Видео и звук при этом "
                     "никуда не отправляются.")
        lay.addWidget(sec)

        # --- сессия ---
        sec = _section("Сессия и исходное состояние")
        self.participant = QLineEdit(s["participant"] or "")
        self.participant.setPlaceholderText("необязательно")
        self.participant.setFixedWidth(220)
        sec.body.addWidget(FieldRow("Участник", "Метка участника для журнала (например, P-01). "
                                                "Не используйте полные имена.", self.participant))
        sec.body.addWidget(_divider())
        self._toggle(sec, "personal_baseline", "Личная база в начале сессии",
                     "Первые 40 с с лицом в кадре фиксируют обычный уровень моргания, взгляда, "
                     "мимики и речи этого человека. Без неё используются типовые значения.")
        lay.addWidget(sec)

        # --- отображение ---
        sec = _section("Отображение")
        self._toggle(sec, "overlay_face_box", "Рамка лица", "Уголки вокруг найденного лица.")
        sec.body.addWidget(_divider())
        self._toggle(sec, "overlay_eyes", "Контуры глаз и радужки",
                     "При закрытых глазах контур подсвечивается красным.")
        sec.body.addWidget(_divider())
        self._toggle(sec, "overlay_mesh", "Точки сетки лица", "478 ключевых точек MediaPipe.")
        sec.body.addWidget(_divider())
        self._toggle(sec, "captions_on_video", "Субтитры поверх видео",
                     "Последняя распознанная фраза внизу изображения.")
        lay.addWidget(sec)

        # --- приватность ---
        sec = _section("Приватность и хранение",
                       "Сохраняются только числовые показатели и текст — без кадров и звука, "
                       "если явно не включено иное.")
        self._toggle(sec, "save_log", "Сохранять журнал показателей",
                     "analysis_log.csv: время, показатели лица, взгляда, мимики, покраснения.")
        sec.body.addWidget(_divider())
        self._toggle(sec, "save_subtitles", "Сохранять текст субтитров", "subtitles.txt и subtitles_live.txt.")
        sec.body.addWidget(_divider())
        self._toggle(sec, "save_video", "Сохранять обработанное видео",
                     "Только для видеофайлов: копия с рамкой лица и контурами глаз. "
                     "Видео с камеры и экрана не записывается никогда.")
        sec.body.addWidget(_divider())
        self._toggle(sec, "protect_window", "Скрывать окно при захвате экрана",
                     "В режиме «Экран» окно программы не попадает в собственный захват.")
        sec.body.addWidget(_divider())
        folder_button = make_button("Открыть папку", "folder", None, "sm")
        folder_button.clicked.connect(lambda: open_path(self.results_dir))
        sec.body.addWidget(FieldRow("Папка результатов", self.results_dir, folder_button))
        lay.addWidget(sec)

        # --- разметка ---
        sec = _section("Разметка эксперимента",
                       "Для исследовательской проверки сигналов скриптом evaluate_lie_signals.py. "
                       "Приложение ничего не определяет само: метку ставит человек вручную.")
        self._toggle(sec, "markup_enabled", "Ручная разметка",
                     "Показывает переключатель меток на нижней панели; клавиши T, L, N.")
        lay.addWidget(sec)
        lay.addStretch(1)

        footer = QHBoxLayout()
        footer.setSpacing(10)
        note = label("Источник, микрофон и модель речи применятся к следующей сессии."
                     if session_active else "", "Hint")
        footer.addWidget(note, 1)
        cancel = make_button("Отмена")
        cancel.clicked.connect(self.reject)
        save = make_button("Сохранить", "check", "primary")
        save.clicked.connect(self.accept)
        footer.addWidget(cancel)
        footer.addWidget(save)
        self.root.addLayout(footer)

    def _toggle(self, section, key, title, hint):
        switch = ToggleSwitch(bool(self.settings[key]))
        self.controls[key] = switch
        section.body.addWidget(FieldRow(title, hint, switch))

    def _run_probe(self):
        self.probe_label.setText("Проверка камеры…")
        self._probe.probe(self.camera.currentData())

    def values(self):
        out = {key: switch.isChecked() for key, switch in self.controls.items()}
        out["camera_index"] = int(self.camera.currentData())
        out["audio_device"] = self.device.currentData()
        out["whisper_model"] = self.model.currentData()
        out["participant"] = self.participant.text().strip()[:40]
        return out


# ---------------------------------------------------------------- история

class _SessionItem(QWidget):
    def __init__(self, record, parent=None):
        super().__init__(parent)
        self.record = record
        self.setFixedHeight(58)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.record
        when = r.started.strftime("%d.%m.%Y  %H:%M") if r.started else r.name
        p.setFont(font(13, 600))
        p.setPen(qc(C.TEXT))
        p.drawText(QRectF(14, 9, self.width() - 28, 20), Qt.AlignLeft | Qt.AlignVCenter, when)
        p.setFont(font(12))
        p.setPen(qc(C.TEXT_3))
        meta = f"{r.source_label} · {r.size_bytes / 1024:.0f} КБ"
        fm = QFontMetrics(p.font())
        p.drawText(QRectF(14, 30, self.width() - 28, 18), Qt.AlignLeft | Qt.AlignVCenter,
                   fm.elidedText(meta, Qt.ElideRight, self.width() - 28))


class _Stat(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.value = "—"
        self.setFixedHeight(64)
        self.setMinimumWidth(120)

    def set_value(self, value):
        self.value = value
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(qc(C.BORDER), 1))
        p.setBrush(qc(C.SURFACE_2))
        p.drawRoundedRect(r, 12, 12)
        p.setFont(font(11, 600))
        p.setPen(qc(C.TEXT_3))
        title = QFontMetrics(p.font()).elidedText(self.title, Qt.ElideRight, int(r.width() - 24))
        p.drawText(QRectF(12, 8, r.width() - 24, 16), Qt.AlignLeft | Qt.AlignVCenter, title)
        p.setFont(numeric_font(20))
        p.setPen(qc(C.TEXT))
        p.drawText(QRectF(12, 28, r.width() - 24, 28), Qt.AlignLeft | Qt.AlignVCenter, self.value)


class HistoryDialog(_DialogBase):
    def __init__(self, parent=None, results_dir="", active_dir=None, saving_enabled=True):
        super().__init__("История сессий",
                         "Сохранённые сессии: только числовые показатели и текст субтитров. "
                         + ("" if saving_enabled else "Сохранение новых сессий сейчас отключено в настройках."),
                         parent, (1060, 700))
        self.results_dir = results_dir
        self.active_dir = os.path.realpath(active_dir) if active_dir else None

        body = QHBoxLayout()
        body.setSpacing(16)
        self.list = QListWidget()
        self.list.setFixedWidth(280)
        self.list.setSpacing(2)
        self.list.currentRowChanged.connect(self._show)
        body.addWidget(self.list)

        self.detail = Card(margins=(22, 18, 22, 16))
        self.detail.body.setSpacing(10)
        self.title = label("", "CardTitle")
        self.meta = label("", "CardSubtitle")
        self.detail.body.addWidget(self.title)
        self.detail.body.addWidget(self.meta)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        self.stats = {}
        for i, (key, title) in enumerate((("duration", "ДЛИТЕЛЬНОСТЬ"), ("face", "ЛИЦО В КАДРЕ"),
                                          ("blink", "МОРГАНИЕ, В МИНУТУ"),
                                          ("rapid", "БЫСТРЫЕ ИЗМЕНЕНИЯ МИМИКИ"),
                                          ("phrases", "РАСПОЗНАННЫХ ФРАЗ"),
                                          ("deviation", "ОТКЛОНЕНИЕ ОТ БАЗЫ, СРЕДНЕЕ"))):
            stat = _Stat(title)
            self.stats[key] = stat
            grid.addWidget(stat, i // 3, i % 3)
        self.detail.body.addLayout(grid)
        self.expressions = label("", "Hint", wrap=True)
        self.detail.body.addWidget(self.expressions)
        self.detail.body.addWidget(label("Субтитры", "SectionTitle"))
        self.transcript = QTextBrowser()
        self.transcript.setOpenLinks(False)
        self.detail.body.addWidget(self.transcript, 1)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.open_folder = make_button("Открыть папку", "folder")
        self.open_folder.clicked.connect(lambda: self._current() and open_path(self._current().path))
        self.open_csv = make_button("Журнал CSV", "external")
        self.open_csv.clicked.connect(lambda: self._current() and open_path(self._current().log_path))
        self.delete = make_button("Удалить", "trash", "danger")
        self.delete.clicked.connect(self._delete)
        buttons.addWidget(self.open_folder)
        buttons.addWidget(self.open_csv)
        buttons.addStretch(1)
        buttons.addWidget(self.delete)
        self.detail.body.addLayout(buttons)
        body.addWidget(self.detail, 1)
        self.root.addLayout(body, 1)

        footer = QHBoxLayout()
        footer.addWidget(label(f"Папка результатов: {results_dir}", "Hint"), 1)
        close = make_button("Закрыть")
        close.clicked.connect(self.accept)
        footer.addWidget(close)
        self.root.addLayout(footer)

        self.records = []
        self.reload()

    def reload(self):
        self.records = list_sessions(self.results_dir)
        self.list.clear()
        for record in self.records:
            item = QListWidgetItem()
            item.setSizeHint(QSize(260, 58))
            self.list.addItem(item)
            self.list.setItemWidget(item, _SessionItem(record))
        if self.records:
            self.list.setCurrentRow(0)
        else:
            self._show(-1)

    def _current(self):
        row = self.list.currentRow()
        return self.records[row] if 0 <= row < len(self.records) else None

    def _show(self, row):
        record = self.records[row] if 0 <= row < len(self.records) else None
        enabled = record is not None
        for widget in (self.open_folder, self.open_csv, self.delete):
            widget.setEnabled(enabled)
        if record is None:
            self.title.setText("Сохранённых сессий пока нет")
            self.meta.setText("Они появятся здесь после завершения анализа, если сохранение включено.")
            for stat in self.stats.values():
                stat.set_value("—")
            self.expressions.setText("")
            self.transcript.setHtml("")
            return
        is_active = self.active_dir is not None and os.path.realpath(record.path) == self.active_dir
        self.delete.setEnabled(not is_active)
        self.open_csv.setEnabled(record.has_log)
        when = record.started.strftime("%d.%m.%Y, %H:%M:%S") if record.started else record.name
        self.title.setText(when + ("  ·  идёт сейчас" if is_active else ""))
        summary = load_summary(record)
        self.meta.setText(f"{record.source_label} · {record.name} · {record.size_bytes / 1024:.0f} КБ")

        def pct(v):
            return "—" if v is None else f"{v * 100:.0f}%"

        self.stats["duration"].set_value(fmt_time(summary.get("duration_s") or 0))
        self.stats["face"].set_value(pct(summary.get("face_share")))
        blink = summary.get("blink_rate_mean")
        self.stats["blink"].set_value("—" if blink is None else f"{blink:.0f}")
        rapid = summary.get("rapid_changes")
        self.stats["rapid"].set_value("—" if rapid is None else str(rapid))
        phrases = summary.get("phrases")
        self.stats["phrases"].set_value("—" if phrases is None else str(phrases))
        self.stats["deviation"].set_value(pct(summary.get("deviation_mean")))
        expr = summary.get("expressions") or []
        if expr:
            parts = [f"{name} {share * 100:.0f}%" for name, share in expr]
            self.expressions.setText("Преобладающие движения лица: " + " · ".join(parts))
        else:
            self.expressions.setText("")
        lines = read_transcript(record)
        if lines:
            body = "".join(f"<p style='margin:0 0 6px 0'>{html.escape(line)}</p>" for line in lines)
        else:
            body = f"<p style='color:{C.TEXT_4}'>Текст субтитров не сохранялся или речи не было.</p>"
        self.transcript.setHtml(f"<div style='font-family:\"{Fonts.ui}\"; font-size:13px'>{body}</div>")

    def _delete(self):
        record = self._current()
        if record is None:
            return
        answer = QMessageBox.question(
            self, "Удалить сессию",
            f"Удалить папку сессии «{record.name}» со всеми файлами?\nЭто действие нельзя отменить.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        try:
            delete_session(record, self.results_dir)
        except Exception as error:
            QMessageBox.warning(self, "Не удалось удалить", str(error))
        self.reload()


# ---------------------------------------------------------------- журнал

class JournalDialog(_DialogBase):
    def __init__(self, parent=None, events=(), phrases=(), system_lines=(), session_dir=None):
        super().__init__("Журнал анализа",
                         "События текущей сессии, распознанная речь и системные сообщения.",
                         parent, (900, 640))
        self.session_dir = session_dir
        self.tabs = QTabWidget()
        self.events = QTreeWidget()
        self.events.setHeaderLabels(["Время", "Тип", "Описание"])
        self.events.setRootIsDecorated(False)
        self.events.setColumnWidth(0, 80)
        self.events.setColumnWidth(1, 120)
        self.events.setUniformRowHeights(True)
        self.tabs.addTab(self.events, "События")
        self.phrases = QTextBrowser()
        self.phrases.setOpenLinks(False)
        self.tabs.addTab(self.phrases, "Субтитры")
        self.system = QPlainTextEdit()
        self.system.setReadOnly(True)
        self.system.setMaximumBlockCount(5000)
        self.system.setFont(font(12))
        self.tabs.addTab(self.system, "Системный журнал")
        self.root.addWidget(self.tabs, 1)

        footer = QHBoxLayout()
        footer.setSpacing(8)
        self.folder_button = make_button("Папка сессии", "folder")
        self.folder_button.clicked.connect(lambda: open_path(self.session_dir))
        self.csv_button = make_button("Журнал CSV", "external")
        self.csv_button.clicked.connect(
            lambda: open_path(os.path.join(self.session_dir, "analysis_log.csv") if self.session_dir else None))
        copy = make_button("Копировать", "copy")
        copy.clicked.connect(self._copy)
        footer.addWidget(self.folder_button)
        footer.addWidget(self.csv_button)
        footer.addWidget(copy)
        footer.addStretch(1)
        close = make_button("Закрыть")
        close.clicked.connect(self.close)
        footer.addWidget(close)
        self.root.addLayout(footer)

        self._phrases = []
        for ev in events:
            self.add_event(ev, scroll=False)
        for t, text in phrases:
            self.add_phrase(t, text)
        for line in system_lines:
            self.system.appendPlainText(line)
        self.set_session_dir(session_dir)
        self.events.scrollToBottom()

    def set_session_dir(self, session_dir):
        self.session_dir = session_dir
        has = bool(session_dir and os.path.isdir(session_dir))
        self.folder_button.setEnabled(has)
        self.csv_button.setEnabled(has and os.path.exists(os.path.join(session_dir, "analysis_log.csv")))

    def add_event(self, ev, scroll=True):
        item = QTreeWidgetItem([fmt_time(ev.t), ev.kind, ev.text])
        item.setData(0, Qt.UserRole, float(ev.t))
        if ev.level in ("warning", "error") or ev.kind == EV_MICRO:
            item.setForeground(2, qc(C.ACCENT_PRESSED))
        if ev.kind == EV_SPEECH:
            item.setForeground(2, qc(C.TEXT_2))
        # фраза приходит после распознавания, а стоит по времени произнесения
        index = self.events.topLevelItemCount()
        while index > 0 and self.events.topLevelItem(index - 1).data(0, Qt.UserRole) > ev.t:
            index -= 1
        self.events.insertTopLevelItem(index, item)
        if scroll:
            self.events.scrollToBottom()

    def add_phrase(self, t, text):
        self._phrases.append((t, text))
        self.phrases.append(f"<p style='margin:0 0 6px 0'><span style='color:{C.TEXT_3}; "
                            f"font-family:\"{Fonts.numeric}\"'>{fmt_time(t)}</span>&nbsp;&nbsp;"
                            f"{html.escape(text)}</p>")

    def add_system_line(self, line):
        self.system.appendPlainText(line)

    def _copy(self):
        index = self.tabs.currentIndex()
        if index == 0:
            lines = []
            for i in range(self.events.topLevelItemCount()):
                item = self.events.topLevelItem(i)
                lines.append("\t".join(item.text(c) for c in range(3)))
            text = "\n".join(lines)
        elif index == 1:
            text = "\n".join(f"[{fmt_time(t)}] {x}" for t, x in self._phrases)
        else:
            text = self.system.toPlainText()
        QGuiApplication.clipboard().setText(text)
