# -*- coding: utf-8 -*-
"""Главное окно: компоновка, состояния сессии и связь с фоновым потоком."""
import os
from collections import deque
from datetime import datetime

from PySide6.QtCore import QCoreApplication, Qt, QTimer
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, QFrame, QHBoxLayout,
                               QMainWindow, QMessageBox, QScrollArea, QVBoxLayout, QWidget)

from app_paths import FACE_MODEL_PATH, RESULTS_DIR
from capture_protect import set_hwnd_capture_excluded
from gui.dialogs import HistoryDialog, JournalDialog, SettingsDialog, open_path
from gui.icons import app_icon
from gui.observations import ObservationsPanel
from gui.session_worker import SessionConfig, SessionThread
from gui.theme import C
from gui.timeline_view import TimelineCard
from gui.transcript import TranscriptCard, fmt_time
from gui.video_stage import VideoStage
from gui.widgets import (Chip, LogoMark, PrivacyIndicators, SegmentedControl, StatusPill, Toast,
                         label, make_button, set_button_variant, vsep)
from models import EV_CALIBRATION, EV_SPEECH, EV_SYSTEM
from session_history import save_summary
from session_timeline import SessionTimeline
from speech_engine import MODELS_DISABLED, MODELS_FAILED, MODELS_LOADING, MODELS_READY

ACTIVE_STATES = ("starting", "running", "paused", "stopping")
QUALITY_WARNINGS = {
    "плохой свет": "Недостаточно света",
    "неравномерный свет": "Неравномерное освещение",
    "размытие": "Изображение размыто",
    "лицо слишком мелкое": "Лицо слишком далеко",
}
SOURCES = [("camera", "Камера", "camera"), ("file", "Видеофайл", "file"), ("screen", "Экран", "screen")]
CALIBRATION_BANNER_SECONDS = 20.0  # дольше баннер не закрывает видео - остаётся предупреждение
VIDEO_FILTER = "Видео (*.mp4 *.avi *.mov *.mkv *.wmv *.webm *.m4v);;Все файлы (*)"


class MainWindow(QMainWindow):
    def __init__(self, settings, speech_models, log_bus, models_bridge):
        super().__init__()
        self.settings = settings
        self.models = speech_models
        self.state = "idle"
        self.thread = None
        self.session_cfg = None
        self.session_dir = None
        self.timeline = SessionTimeline()
        self.events = []
        self.phrases = []
        self.system_lines = deque(maxlen=3000)
        self.journal = None
        self.file_path = ""
        self.current_source = "camera"
        self._error = None
        self._calib_since = None
        self._protected = False
        self._models_reload_pending = False
        self._was_maximized = False
        self.toast_bottom_margin = 92

        self.setWindowTitle("All the girls")
        self.setWindowIcon(app_icon())
        self._build_ui()
        self._wire_shortcuts()
        self.toast = Toast(self)

        log_bus.line.connect(self._on_log_line)
        models_bridge.changed.connect(self._on_models_changed)

        self._apply_display_settings()
        source = settings["source"] if settings["source"] in ("camera", "file", "screen") else "camera"
        if source == "file":
            source = "camera"  # путь к файлу не запоминаем между запусками
        self.set_source(source, ask_file=False)
        self._update_markup_visibility()
        self._refresh_models_info()
        self._set_state("idle")

        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(250)
        self.ui_timer.timeout.connect(self._on_ui_timer)
        self.ui_timer.start()

    # =====================================================
    # компоновка
    # =====================================================

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("Central")
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_top_bar())

        body = QHBoxLayout()
        body.setContentsMargins(16, 16, 16, 14)
        body.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(14)
        self.stage = VideoStage()
        self.stage.start_requested.connect(self.start_session)
        self.stage.source_selected.connect(self._on_source_clicked)
        self.stage.action_requested.connect(self._on_stage_action)
        left.addWidget(self.stage, 1)
        self.timeline_card = TimelineCard(self.timeline)
        self.timeline_card.setFixedHeight(220)  # подстраивается под окно в resizeEvent
        left.addWidget(self.timeline_card, 0)
        body.addLayout(left, 1)

        right_host = QWidget()
        right = QVBoxLayout(right_host)
        right.setContentsMargins(0, 0, 4, 0)
        right.setSpacing(14)
        self.observations = ObservationsPanel()
        right.addWidget(self.observations, 0)
        self.transcript = TranscriptCard()
        self.transcript.setMinimumHeight(150)
        self.transcript.open_journal.connect(self.open_journal)
        right.addWidget(self.transcript, 1)
        scroll = QScrollArea()
        scroll.setWidget(right_host)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(404)
        body.addWidget(scroll, 0)

        root.addLayout(body, 1)
        root.addWidget(self._build_control_bar())
        self.setCentralWidget(central)

    def _build_top_bar(self):
        bar = QFrame()
        bar.setObjectName("TopBar")
        bar.setFixedHeight(60)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(20, 0, 14, 0)
        lay.setSpacing(12)

        lay.addWidget(LogoMark(28))
        lay.addWidget(label("All the girls", "AppName"))
        lay.addSpacing(6)
        lay.addWidget(vsep(24))
        lay.addSpacing(2)
        self.status_pill = StatusPill()
        lay.addWidget(self.status_pill)
        lay.addSpacing(8)

        self.source_seg = SegmentedControl(SOURCES, height=36)
        self.source_seg.setToolTip("Источник видео")
        self.source_seg.changed.connect(self._on_source_clicked)
        lay.addWidget(self.source_seg)
        self.source_label = label("", "Muted")
        self.source_label.setMaximumWidth(220)
        lay.addWidget(self.source_label)
        lay.addStretch(1)

        session = QWidget()
        session_lay = QVBoxLayout(session)
        session_lay.setContentsMargins(0, 0, 0, 0)
        session_lay.setSpacing(0)
        session_lay.addWidget(label("СЕССИЯ", "Caption"))
        self.clock = label("—", "Clock")
        session_lay.addWidget(self.clock)
        lay.addWidget(session)
        self.participant_chip = Chip("", "user")
        self.participant_chip.setToolTip("Участник (меняется в настройках)")
        lay.addWidget(self.participant_chip)
        lay.addSpacing(4)
        lay.addWidget(vsep(24))
        self.privacy = PrivacyIndicators()
        lay.addWidget(self.privacy)
        settings_button = make_button("", "settings", "ghost", tooltip="Настройки (Ctrl+,)")
        settings_button.clicked.connect(self.open_settings)
        lay.addWidget(settings_button)
        return bar

    def _build_control_bar(self):
        bar = QFrame()
        bar.setObjectName("ControlBar")
        bar.setFixedHeight(72)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(20, 0, 20, 0)
        lay.setSpacing(10)

        self.primary_button = make_button("Начать анализ", "play", "primary", "lg",
                                          tooltip="Начать анализ / пауза (Пробел)")
        self.primary_button.setMinimumWidth(196)
        self.primary_button.clicked.connect(self._on_primary)
        lay.addWidget(self.primary_button)
        self.stop_button = make_button("Завершить", "stop", None, "lg",
                                       tooltip="Завершить сессию и сохранить итоги (Ctrl+E)")
        self.stop_button.clicked.connect(self.stop_session)
        lay.addWidget(self.stop_button)
        lay.addSpacing(4)
        lay.addWidget(vsep(28))
        lay.addSpacing(4)
        self.recal_button = make_button("Перекалибровать", "recalibrate", None, "lg",
                                        tooltip="Заново зафиксировать нейтральное лицо, цвет кожи "
                                                "и личную базу (Ctrl+R)")
        self.recal_button.clicked.connect(self.recalibrate)
        lay.addWidget(self.recal_button)

        self.markup_host = QWidget()
        markup_lay = QHBoxLayout(self.markup_host)
        markup_lay.setContentsMargins(10, 0, 0, 0)
        markup_lay.setSpacing(8)
        markup_lay.addWidget(label("Разметка", "Muted"))
        self.markup = SegmentedControl([("", "Нет", None), ("правда", "Правда · T", None),
                                        ("ложь", "Ложь · L", None)], height=34)
        self.markup.setToolTip("Ручная разметка эксперимента (T, L, N). Приложение само ничего не "
                               "определяет — метка нужна для evaluate_lie_signals.py.")
        self.markup.changed.connect(self._on_markup)
        markup_lay.addWidget(self.markup)
        lay.addWidget(self.markup_host)

        lay.addStretch(1)
        self.hint = label("Пробел — старт и пауза", "Hint")
        lay.addWidget(self.hint)
        lay.addSpacing(6)
        for text, icon_name, slot, tip in (
                ("Журнал", "journal", self.open_journal, "Подробный журнал анализа (Ctrl+J)"),
                ("История", "history", self.open_history, "Сохранённые сессии (Ctrl+H)"),
                ("Настройки", "settings", self.open_settings, "Настройки (Ctrl+,)")):
            button = make_button(text, icon_name, "ghost", "lg", tooltip=tip)
            button.clicked.connect(slot)
            lay.addWidget(button)
        return bar

    def _wire_shortcuts(self):
        def add(seq, slot):
            shortcut = QShortcut(QKeySequence(seq), self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(slot)

        add("Space", self._on_primary)
        add("Ctrl+E", self.stop_session)
        add("Ctrl+R", self.recalibrate)
        add("C", lambda: self._recalibrate_part(True, False, False, "Калибровка мимики"))
        add("R", lambda: self._recalibrate_part(False, True, False, "Калибровка покраснения"))
        add("B", lambda: self._recalibrate_part(False, False, True, "Сбор личной базы"))
        add("T", lambda: self._hotkey_label("правда"))
        add("L", lambda: self._hotkey_label("ложь"))
        add("N", lambda: self._hotkey_label(""))
        add("H", self._toggle_overlays)
        add("F11", self._toggle_fullscreen)
        add("F", self._toggle_fullscreen)
        add("Ctrl+H", self.open_history)
        add("Ctrl+J", self.open_journal)
        add("Ctrl+,", self.open_settings)

    # =====================================================
    # состояние
    # =====================================================

    def _set_state(self, state, text=""):
        self.state = state
        pill = {
            "idle": ("idle", "Готово к работе"),
            "starting": ("busy", text or "Подготовка…"),
            "running": ("live", "Анализ идёт"),
            "paused": ("paused", "Пауза"),
            "stopping": ("busy", "Завершение…"),
            "finished": ("done", "Сессия завершена"),
            "error": ("error", "Ошибка"),
        }[state]
        self.status_pill.set_state(*pill)

        if state == "running":
            set_button_variant(self.primary_button, "dark", "pause", "Пауза")
        elif state == "paused":
            set_button_variant(self.primary_button, "primary", "play", "Продолжить")
        elif state == "finished":
            set_button_variant(self.primary_button, "primary", "play", "Новая сессия")
        elif state == "error":
            set_button_variant(self.primary_button, "primary", "play", "Начать снова")
        else:
            set_button_variant(self.primary_button, "primary", "play", "Начать анализ")
        self.primary_button.setEnabled(state in ("idle", "running", "paused", "finished", "error"))
        self.stop_button.setEnabled(state in ("starting", "running", "paused"))
        self.recal_button.setEnabled(state == "running")
        self.markup.setEnabled(state in ("running", "paused"))

        if state == "idle":
            self.stage.set_mode("empty")
            self.observations.reset()
            self._refresh_speech_idle()
            self.clock.setText("—")
            for key in ("camera", "mic", "analysis", "saving"):
                self.privacy.set_state(key, False)
        elif state == "starting":
            self.stage.set_mode("connecting", text or "Подготовка…")
        elif state == "running":
            self.stage.set_mode("live")
            self.privacy.set_state("analysis", True)
        elif state == "paused":
            self.stage.set_mode("paused")
            self.privacy.set_state("analysis", False)
        elif state in ("finished", "error"):
            self.stage.set_mode(state)
            for key in ("camera", "mic", "analysis", "saving"):
                self.privacy.set_state(key, False)
            self._refresh_speech_idle()
        self.hint.setText({"running": "Пробел — пауза", "paused": "Пробел — продолжить"}
                          .get(state, "Пробел — старт"))

    # =====================================================
    # источник
    # =====================================================

    def _on_source_clicked(self, key):
        if self.state in ACTIVE_STATES:
            if key == self.current_source and key != "file":
                return
            answer = QMessageBox.question(
                self, "Сменить источник",
                "Текущая сессия будет завершена, её результаты сохранятся. Продолжить?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                self.source_seg.set_key(self.current_source, emit=False)
                return
            self.stop_session()
        self.set_source(key, ask_file=True)

    def set_source(self, key, ask_file=True):
        if key == "file" and ask_file:
            if not self._choose_file():
                self.source_seg.set_key(self.current_source, emit=False)
                return
        self.current_source = key
        self.source_seg.set_key(key, emit=False)
        self.settings.set("source", key)
        self.settings.save()
        self._update_source_label()
        if self.state in ("finished", "error"):
            self._set_state("idle")

    def _choose_file(self):
        start_dir = self.settings["last_video_dir"] or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(self, "Выберите видеофайл", start_dir, VIDEO_FILTER)
        if not path:
            return False
        self.file_path = path
        self.settings.set("last_video_dir", os.path.dirname(path))
        return True

    def _update_source_label(self):
        s = self.settings
        key = self.current_source
        if key == "file":
            name = os.path.basename(self.file_path) if self.file_path else "файл не выбран"
            self.source_label.setText(name)
            self.source_label.setToolTip(self.file_path)
            note = f"Файл: {name}" if self.file_path else "Выберите видеофайл"
        elif key == "screen":
            self.source_label.setText("Основной монитор")
            self.source_label.setToolTip("")
            note = "Основной монитор · окно программы скрывается из захвата" if s["protect_window"] \
                else "Основной монитор"
        else:
            self.source_label.setText(f"Камера {s['camera_index']}")
            self.source_label.setToolTip("Номер камеры меняется в настройках")
            note = f"Камера {s['camera_index']} · изменить можно в настройках"
        self.stage.empty.set_source(key, note)
        participant = (s["participant"] or "").strip()
        self.participant_chip.set(participant or "без участника", "neutral" if participant else "muted")

    # =====================================================
    # сессия
    # =====================================================

    def _on_primary(self):
        if self.state in ("idle", "finished", "error"):
            self.start_session()
        elif self.state == "running":
            self.pause()
        elif self.state == "paused":
            self.resume()

    def start_session(self):
        if self.thread is not None:
            if not self.thread.isFinished():
                return
            self.thread = None  # сигнал finished прежнего потока ещё в очереди
        kind = self.current_source
        if kind == "file" and not (self.file_path and os.path.exists(self.file_path)):
            if not self._choose_file():
                return
            self._update_source_label()
        s = self.settings
        if s["speech_enabled"] and self.models.state in (MODELS_DISABLED, MODELS_FAILED) \
                and not self.models.loading:
            self._load_models()
        cfg = SessionConfig(
            source_kind=kind,
            camera_index=int(s["camera_index"] or 0),
            file_path=self.file_path,
            participant=(s["participant"] or "").strip(),
            save_log=bool(s["save_log"]),
            save_subtitles=bool(s["save_subtitles"]),
            save_video=bool(s["save_video"]),
            pace_video_file=bool(s["pace_video_file"]),
            speech_enabled=bool(s["speech_enabled"]),
            audio_device=s["audio_device"],
            personal_baseline=bool(s["personal_baseline"]),
            with_mesh=bool(s["overlay_mesh"]),
            results_dir=RESULTS_DIR,
            face_model_path=FACE_MODEL_PATH,
        )
        self.session_cfg = cfg
        self.session_dir = None
        self._error = None
        self._calib_since = None
        self.timeline.clear()
        self.events = []
        self.phrases = []
        self.transcript.clear()
        self.observations.reset("подготовка…")
        self.markup.set_key("", animate=False)
        self.stage.set_caption("")
        self.stage.set_banner(None)
        self.stage.set_warnings([])
        self.stage.set_progress(None)
        self.stage.set_status("", "", "", "")
        self.stage.image = None
        if self.journal is not None:
            self.journal.close()

        thread = SessionThread(cfg, self.models, self)
        thread.frame_ready.connect(self._on_frame_ready)
        thread.tick.connect(self._on_tick)
        thread.event_emitted.connect(self._on_event)
        thread.phase_changed.connect(self._on_phase)
        thread.failed.connect(self._on_failed)
        thread.session_finished.connect(self._on_finished)
        thread.finished.connect(self._on_thread_finished)
        self.thread = thread
        self.timeline_card.view.live = True
        if self.timeline_card.view.window_s == 0:
            self.timeline_card.set_window(180)
        self._set_state("starting", "Подготовка…")
        if kind == "screen" and s["protect_window"]:
            self._protect(True)
        thread.start()

    def pause(self):
        if self.thread is not None and self.state == "running":
            self.thread.set_paused(True)
            self._set_state("paused")

    def resume(self):
        if self.thread is not None and self.state == "paused":
            self.thread.set_paused(False)
            self._set_state("running")

    def stop_session(self):
        if self.thread is None or self.state not in ("starting", "running", "paused"):
            return
        self._set_state("stopping")
        self.thread.request_stop()

    def recalibrate(self):
        self._recalibrate_part(True, True, True, None)

    def _recalibrate_part(self, mimic, redness, baseline, title):
        if self.thread is None or self.state != "running":
            return
        self.thread.recalibrate(mimic, redness, baseline)
        if title:
            self.toast.show_message(f"{title}: смотрите в камеру, лицо нейтральное")

    def _hotkey_label(self, value):
        if self.settings["markup_enabled"] and self.state in ("running", "paused"):
            self.markup.set_key(value)
            self._on_markup(value)

    def _on_markup(self, value):
        if self.thread is not None:
            self.thread.set_label(value)

    def _protect(self, enabled):
        if enabled == self._protected:
            return
        if set_hwnd_capture_excluded(int(self.winId()), enabled) or not enabled:
            self._protected = enabled

    # =====================================================
    # сигналы потока
    # =====================================================

    def _on_frame_ready(self):
        if self.thread is None:
            return
        item = self.thread.take_frame()
        if item is not None:
            self.stage.set_frame(*item)

    def _on_tick(self, tick):
        if self.thread is None or self.sender() is not self.thread:
            return
        if tick.session_dir and not self.session_dir:
            self.session_dir = tick.session_dir
            if self.journal is not None:
                self.journal.set_session_dir(self.session_dir)
        s = tick.snapshot
        if s is not None and not tick.paused:
            self.timeline.add_snapshot(s)
        self.observations.update_from(tick, self.timeline, paused=tick.paused)

        cfg = self.session_cfg
        is_file = cfg.source_kind == "file"
        badge = "ПАУЗА" if tick.paused else ("ФАЙЛ" if is_file else ("ЭКРАН" if cfg.source_kind == "screen" else "LIVE"))
        if is_file and tick.video_duration:
            time_text = f"{fmt_time(tick.video_pos or 0)} / {fmt_time(tick.video_duration)}"
            self.stage.set_progress((tick.video_pos or 0) / tick.video_duration)
        else:
            time_text = fmt_time(tick.t)
            self.stage.set_progress(None)
        fps_text = f"{tick.fps:.0f} к/с · {tick.proc_ms:.0f} мс" if (tick.fps and not tick.paused) else ""
        self.stage.set_status(badge, cfg.source_label, time_text, fps_text)

        self.stage.set_caption(tick.live_text, live=True)
        self.transcript.set_live(tick.live_text)
        self._update_speech_chip(tick)

        warnings = []
        if s is not None and not tick.paused:
            if not s.face_model_ok:
                warnings.append("Анализ лица недоступен")
            elif not s.face_detected:
                if s.t > 1.0:
                    warnings.append("Лицо не найдено")
            elif s.quality is not None and s.quality.note in QUALITY_WARNINGS:
                warnings.append(QUALITY_WARNINGS[s.quality.note])

        banner = None
        calibrating = (s is not None and s.face_detected and not tick.paused
                       and (not s.mimic_ready or not s.redness_ready))
        if calibrating:
            if self._calib_since is None:
                self._calib_since = s.t
            if s.t - self._calib_since < CALIBRATION_BANNER_SECONDS:
                progress = (s.mimic_progress + s.redness_progress) / 2.0
                banner = ("Калибровка исходного состояния",
                          "Смотрите в камеру, лицо нейтральное, свет ровный", progress)
            elif not s.redness_ready:
                warnings.append("Цвет кожи не калибруется — проверьте освещение")
        elif s is not None and s.mimic_ready and s.redness_ready:
            self._calib_since = None
        self.stage.set_warnings(warnings)
        self.stage.set_banner(banner)

        self.privacy.set_state("camera", tick.camera_active,
                               "камера открыта" + (" (идёт только просмотр)" if tick.paused else ""))
        self.privacy.set_state("mic", tick.mic_active, "микрофон слушает только во время анализа")
        self.privacy.set_state("analysis", not tick.paused)
        self.privacy.set_state("saving", tick.saving, self.session_dir or "")
        self.clock.setText(fmt_time(tick.t))

    def _update_speech_chip(self, tick):
        state = tick.speech_state
        tone = {"слушаю": "accent", "нет микрофона": "accent", "недоступно": "accent"}.get(state, "muted")
        self.transcript.set_status(state, tone)

    def _on_event(self, ev):
        if self.thread is None:
            return
        self.events.append(ev)
        self.timeline.add_event(ev)
        if ev.kind == EV_SPEECH:
            self.phrases.append((ev.t, ev.text))
            self.transcript.add_phrase(ev.t, ev.text)
            if self.journal is not None:
                self.journal.add_phrase(ev.t, ev.text)
        if self.journal is not None:
            self.journal.add_event(ev)
        if ev.kind == EV_CALIBRATION and ev.level == "notice" and "собрана" in ev.text:
            self.toast.show_message(ev.text, "success")
        elif ev.kind == EV_SYSTEM and ev.level in ("error", "warning"):
            self.toast.show_message(ev.text, ev.level)

    def _on_phase(self, phase, text):
        if self.sender() is not self.thread:
            return
        if phase == "opening":
            self._set_state("starting", text)
        elif phase == "running" and self.state in ("starting", "paused"):
            self._set_state("running")
        elif phase == "paused" and self.state == "running":
            self._set_state("paused")
        elif phase == "stopping" and self.state not in ("stopping", "error"):
            self._set_state("stopping")

    def _on_failed(self, title, detail):
        self._error = (title, detail)
        self.toast.show_message(title, "error", 4500)

    def _on_finished(self, result):
        self._protect(False)
        self.timeline_card.view.live = False
        self.timeline_card.view.update()
        self.session_dir = result.session_dir or self.session_dir
        if self.journal is not None:
            self.journal.set_session_dir(self.session_dir)

        if self._error and not self.timeline.t:
            title, detail = self._error
            self._set_state("error")
            self.stage.show_card("error", "alert", title, detail,
                                 actions=[("retry", "Повторить", "recalibrate"),
                                          ("settings", "Настройки", "settings")])
            return
        if result.reason == "cancelled" or not self.timeline.t:
            self._set_state("idle")  # отменено до первого кадра - итогов нет
            return

        summary = self.timeline.summary()
        summary.update({
            "app": "All the girls",
            "source": result.source_label,
            "source_kind": result.source_kind,
            "participant": result.participant,
            "started_at": result.started_at.isoformat(timespec="seconds"),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "end_reason": result.reason,
            "note": "Показатели описывают наблюдаемые изменения лица, взгляда и речи и не "
                    "определяют эмоции, намерения или правдивость человека.",
        })
        saved = save_summary(self.session_dir, summary) if self.session_dir else None

        self._set_state("finished" if result.reason != "error" else "error")
        self.observations.mute("последнее значение сессии")
        self.transcript.set_live("")
        title = {"eof": "Видео обработано", "error": "Сессия прервана"}.get(result.reason, "Сессия завершена")
        parts = [f"Длительность {fmt_time(summary['duration_s'])}."]
        if self._error:
            parts.append(f"{self._error[0]}: {self._error[1]}")
        if saved:
            rel = os.path.relpath(self.session_dir, os.path.dirname(RESULTS_DIR))
            parts.append(f"Показатели и текст сохранены в папке <b>{rel}</b>.")
        else:
            parts.append("История не сохранялась — сохранение отключено в настройках.")
        if result.output_video:
            parts.append("Обработанное видео сохранено рядом с журналом.")

        def num(v, fmt="{:.0f}"):
            return "—" if v is None else fmt.format(v)

        face = summary.get("face_share")
        stats = [
            ("Лицо в кадре", num(None if face is None else face * 100), "%"),
            ("Моргание", num(summary.get("blink_rate_mean")), "/мин"),
            ("Быстрых изм.", str(summary.get("rapid_changes", 0)), ""),
            ("Фраз", str(summary.get("phrases", 0)), ""),
        ]
        actions = [("new", "Новая сессия", "play"), ("history", "История", "history")]
        if self.session_dir:
            actions.append(("folder", "Папка", "folder"))
        self.stage.show_card("error" if result.reason == "error" else "info",
                             "alert" if result.reason == "error" else "check",
                             title, " ".join(parts), stats, actions)
        if self.timeline.duration > 600:
            self.timeline_card.set_window(0)

        if self._models_reload_pending:
            self._models_reload_pending = False
            self._load_models()

    def _on_thread_finished(self):
        thread = self.sender()
        if thread is self.thread:
            self.thread = None
        if thread is not None:
            thread.deleteLater()

    def _on_stage_action(self, key):
        if key in ("new", "retry"):
            self.start_session()
        elif key == "history":
            self.open_history()
        elif key == "folder":
            open_path(self.session_dir)
        elif key == "settings":
            self.open_settings()

    # =====================================================
    # модели речи
    # =====================================================

    def _load_models(self):
        s = self.settings
        if not s["speech_enabled"]:
            self.models.disable()
            self._refresh_models_info()
            return
        from gui.app import start_model_loading

        start_model_loading(self.models, s)
        self._refresh_models_info()

    def _on_models_changed(self):
        self._refresh_models_info()
        if self.models.state == MODELS_FAILED:
            self.toast.show_message("Распознавание речи недоступно: " + self.models.error, "warning", 5000)

    def _refresh_models_info(self):
        face_ok = os.path.exists(FACE_MODEL_PATH)
        dot_ok = f"<span style='color:{C.ON_DARK_2}'>●</span>"
        dot_bad = f"<span style='color:{C.ACCENT}'>●</span>"
        size = self.settings["whisper_model"]
        state = self.models.state
        if not self.settings["speech_enabled"]:
            speech = f"{dot_ok} Речь · выключена"
        elif state == MODELS_READY:
            speech = f"{dot_ok} Речь · Whisper {self.models.model_size} готов"
            if self.models.correction_active:
                speech += " · коррекция текста"
        elif state == MODELS_LOADING:
            speech = f"{dot_ok} Речь · загрузка Whisper {size}…"
        elif state == MODELS_FAILED:
            speech = f"{dot_bad} Речь · недоступна"
        else:
            speech = f"{dot_ok} Речь · не загружена"
        face = f"{dot_ok} Модель лица · готова" if face_ok else f"{dot_bad} Модель лица · нет файла"
        self.stage.empty.set_models_line(f"{face}&nbsp;&nbsp;&nbsp;&nbsp;{speech}")
        if self.state not in ACTIVE_STATES:
            self._refresh_speech_idle()

    def _refresh_speech_idle(self):
        if not self.settings["speech_enabled"]:
            self.observations.set_speech_idle("Выключено", "можно включить в настройках")
            self.transcript.set_status("выключено", "muted")
            return
        state = self.models.state
        if state == MODELS_READY:
            self.observations.set_speech_idle("Готово", f"Whisper {self.models.model_size} загружен")
            self.transcript.set_status("готово", "muted")
        elif state == MODELS_LOADING:
            self.observations.set_speech_idle("Загрузка", "модель Whisper загружается…")
            self.transcript.set_status("загрузка", "muted")
        elif state == MODELS_FAILED:
            self.observations.set_speech_idle("Недоступно", self.models.error or "ошибка загрузки", "attention")
            self.transcript.set_status("недоступно", "accent")
        else:
            self.observations.set_speech_idle(None, "модель не загружена")
            self.transcript.set_status("выключено", "muted")

    # =====================================================
    # диалоги
    # =====================================================

    def open_settings(self):
        dialog = SettingsDialog(self.settings, self, session_active=self.state in ACTIVE_STATES,
                                results_dir=RESULTS_DIR)
        if dialog.exec() != QDialog.Accepted:
            return
        old = dict(self.settings.data)
        self.settings.update(dialog.values())
        self.settings.save()
        self._apply_display_settings()
        self._update_markup_visibility()
        self._update_source_label()
        if self.thread is not None:
            self.thread.set_mesh(bool(self.settings["overlay_mesh"]))
        speech_keys = ("speech_enabled", "whisper_model", "text_correction", "allow_model_download")
        if any(old.get(k) != self.settings[k] for k in speech_keys):
            if self.state in ACTIVE_STATES:
                self._models_reload_pending = True
                self.toast.show_message("Новые настройки речи применятся после завершения сессии")
            else:
                self._load_models()
        self._refresh_models_info()
        self.toast.show_message("Настройки сохранены", "success", 2000)

    def open_history(self):
        saving = bool(self.settings["save_log"] or self.settings["save_subtitles"])
        active = self.session_dir if self.state in ACTIVE_STATES else None
        HistoryDialog(self, RESULTS_DIR, active, saving).exec()

    def open_journal(self):
        if self.journal is not None:
            self.journal.raise_()
            self.journal.activateWindow()
            return
        self.journal = JournalDialog(self, self.events, self.phrases, list(self.system_lines),
                                     self.session_dir)
        self.journal.setAttribute(Qt.WA_DeleteOnClose)
        self.journal.destroyed.connect(self._journal_closed)
        self.journal.show()

    def _journal_closed(self, *args):
        self.journal = None

    # =====================================================
    # прочее
    # =====================================================

    def _apply_display_settings(self):
        s = self.settings
        self.stage.show_box = bool(s["overlay_face_box"])
        self.stage.show_eyes = bool(s["overlay_eyes"])
        self.stage.show_mesh = bool(s["overlay_mesh"])
        self.stage.show_captions = bool(s["captions_on_video"])
        self.stage.update()

    def _update_markup_visibility(self):
        self.markup_host.setVisible(bool(self.settings["markup_enabled"]))

    def _toggle_overlays(self):
        hidden = not (self.stage.show_box or self.stage.show_eyes or self.stage.show_mesh)
        self.stage.show_box = hidden
        self.stage.show_eyes = hidden
        self.stage.show_mesh = hidden and bool(self.settings["overlay_mesh"])
        self.stage.update()
        self.toast.show_message("Подсветка лица " + ("включена" if hidden else "скрыта"), "info", 1500)

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showMaximized() if self._was_maximized else self.showNormal()
        else:
            self._was_maximized = self.isMaximized()
            self.showFullScreen()

    def _on_log_line(self, line):
        stamped = f"{datetime.now():%H:%M:%S}  {line}"
        self.system_lines.append(stamped)
        if self.journal is not None:
            self.journal.add_system_line(stamped)

    def _on_ui_timer(self):
        self.timeline_card.view.refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # видео - главный элемент: шкала занимает ~30 % высоты левой колонки
        column = self.height() - 60 - 72 - 30 - 14
        self.timeline_card.setFixedHeight(int(max(150, min(260, column * 0.3))))
        if hasattr(self, "toast"):
            self.toast.reposition()

    def closeEvent(self, event):
        if self.thread is not None:
            self.thread.request_stop()
            QApplication.setOverrideCursor(Qt.WaitCursor)
            done = self.thread.wait(8000)
            QApplication.restoreOverrideCursor()
            QCoreApplication.processEvents()  # дописать итоги сессии
            if not done:
                print("Поток сессии не завершился вовремя.")
        self._protect(False)
        self.settings.save()
        event.accept()


def initial_geometry(window):
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        window.resize(1440, 900)
        return
    area = screen.availableGeometry()
    width = min(max(int(area.width() * 0.92), 1180), 1680, area.width())
    height = min(max(int(area.height() * 0.92), 720), 1040, area.height())
    window.resize(width, height)
    window.move(area.x() + (area.width() - width) // 2, area.y() + (area.height() - height) // 2)
    window.setMinimumSize(min(1100, area.width()), min(680, area.height()))
