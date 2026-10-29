# -*- coding: utf-8 -*-
"""All the girls — точка входа.

    python app.py             настольный интерфейс (PySide6)
    python app.py --classic   прежний интерфейс в окне OpenCV (выбор источника в консоли)

Анализ кадра (analysis_pipeline) и распознавание речи (speech_engine) общие для
обоих режимов.
"""
import argparse
import os
import sys
import textwrap
import time

import cv2

from analysis_logger import AnalysisLogger
from analysis_pipeline import AnalysisPipeline
from app_paths import FACE_MODEL_PATH, PANEL_LAYOUT_FILE, make_session_dir
from audio_module import AudioModule
from capture_protect import exclude_window_from_capture
from models import LABEL_LIE, LABEL_TRUTH
from speech_engine import SpeechModels, SpeechSession, load_file_audio
from subtitle_storage import SubtitleStorage
from ui_widgets import (GRAY, GREEN, RED, WHITE, DisplayMapper, PanelManager,
                        draw_bar, load_font, new_panel, to_bgra)
from video_source import ScreenSource, VideoFileSource, choose_video_source

CANVAS_W, CANVAS_H = 1280, 720

# Для видеофайла: воспроизводить в реальном времени, чтобы субтитры (Whisper)
# успевали за звуком. False - обрабатывать как можно быстрее (субтитры отстанут).
PACE_VIDEO_FILE = True

PANEL_RENDER_PERIOD = 2  # раз в сколько кадров перерисовывать панели


def session_base(cap):
    if isinstance(cap, VideoFileSource):
        return os.path.splitext(os.path.basename(cap.filepath))[0]
    if isinstance(cap, ScreenSource):
        return "screen"
    return "webcam"


class App:
    """Классический режим: панели рисуются поверх кадра в окне OpenCV."""

    # Заголовок окна только латиницей: OpenCV на Windows плохо работает
    # с кириллицей в названии окна, а по нему ищется окно для защиты от захвата.
    WINDOW_TITLE = "Mimic Analyzer"

    def __init__(self, cap, participant=""):
        cv2.setUseOptimized(True)

        self.cap = cap
        self.participant = participant
        self.is_screen_mode = isinstance(cap, ScreenSource)
        self.is_video_file = isinstance(cap, VideoFileSource)
        self.window_protected = False
        self.protect_attempts = 0

        self.session_dir = make_session_dir(session_base(cap))
        print("Результаты сеанса:", self.session_dir)

        # время на таймлайне (секунды): для файла - время кадра, для камеры - с начала сеанса
        self.t0 = time.time()
        self.video_time = 0.0

        self.video_writer = None
        self.output_video_path = None

        # --- звук и субтитры ---
        self.audio = AudioModule()
        self.speech_models = SpeechModels()
        self.speech_models.load(model_size="small", use_correction=True)
        self.storage = SubtitleStorage(
            live_file=os.path.join(self.session_dir, "subtitles_live.txt"),
            final_file=os.path.join(self.session_dir, "subtitles.txt"),
            time_fn=lambda: self.video_time,
        )
        self.speech = None
        if self.speech_models.ready:
            self.speech = SpeechSession(self.speech_models, self.audio, self.storage,
                                        time_fn=lambda: self.video_time)

        self.file_audio = None
        self.file_audio_pos = 0
        if self.is_video_file:
            print("Читаю звук из видеофайла...")
            self.file_audio = load_file_audio(cap.filepath, self.audio.samplerate)
            if self.file_audio is None:
                print("Звука в файле нет - субтитров не будет.")

        # --- анализ ---
        self.pipeline = AnalysisPipeline(
            model_path=FACE_MODEL_PATH,
            logger=AnalysisLogger(filepath=os.path.join(self.session_dir, "analysis_log.csv"),
                                  min_interval_sec=0.7),
            participant=participant,
        )

        # --- интерфейс ---
        self.panels = PanelManager(CANVAS_W, CANVAS_H, layout_file=PANEL_LAYOUT_FILE)
        self.panels.register("volume", (15, 8))
        self.panels.register("eyes", (15, 50))
        self.panels.register("redness", (15, 305))
        self.panels.register("mimic", (CANVAS_W - 435, 50))
        self.panels.register("suspicion", (CANVAS_W - 435, 345))
        self.panels.register("subtitles", ((CANVAS_W - 560) // 2, CANVAS_H - 82 - 12))
        self.panels.register("label", (15, 600))
        self.mapper = DisplayMapper(self.WINDOW_TITLE, CANVAS_W, CANVAS_H)
        self.fullscreen = False
        self.frame_index = 0

    # =====================================================
    # ВРЕМЯ И ЗВУК ИЗ ФАЙЛА
    # =====================================================

    def timeline(self):
        if self.is_video_file:
            return self.cap.get_time()
        return time.time() - self.t0

    def feed_file_audio(self, t):
        if self.file_audio is None or self.speech is None:
            return
        end = min(int(t * self.audio.samplerate), len(self.file_audio))
        while self.file_audio_pos < end:
            nxt = min(self.file_audio_pos + self.audio.blocksize, end)
            self.audio.feed(self.file_audio[self.file_audio_pos:nxt])
            self.file_audio_pos = nxt

    # =====================================================
    # ОТРИСОВКА ПАНЕЛЕЙ-ИЗОБРАЖЕНИЙ
    # =====================================================

    @staticmethod
    def render_volume(volume):
        img, draw = new_panel(230, 34)
        draw_bar(draw, 10, 9, min(volume * 14.0, 1.0), "Звук", GREEN, load_font(15),
                 label_w=50, width=150, show_value=False)
        return to_bgra(img)

    @staticmethod
    def render_subtitles(lines):
        if not lines:
            return None
        wrapped = []
        for line in lines:
            wrapped.extend(textwrap.wrap(line, width=46) or [""])
        wrapped = wrapped[-2:]

        img, draw = new_panel(560, 82, outline=False)
        font = load_font(18)
        y = 8
        for line in wrapped:
            draw.text((14, y), line, font=font, fill=WHITE)
            y += 24
        return to_bgra(img)

    def set_label(self, label):
        if label == self.pipeline.label:
            return
        self.pipeline.set_label(label, self.video_time)
        print(f"[{self.video_time:6.1f} c] разметка: {label or 'снята'}")

    def render_label(self):
        img, draw = new_panel(330, 80)
        font, small = load_font(17), load_font(15)
        text, color = {"": ("нет", GRAY), LABEL_TRUTH: ("ПРАВДА", GREEN),
                       LABEL_LIE: ("ЛОЖЬ", RED)}[self.pipeline.label]
        draw.text((12, 8), f"Разметка: {text}", font=font, fill=color)
        draw.text((12, 34), "t - правда   l - ложь   n - снять", font=small, fill=GRAY)
        suspicion = self.pipeline.last_suspicion
        if suspicion is None or not suspicion.baseline_ready:
            draw.text((12, 56), "база собирается - пока не размечайте", font=small, fill=GRAY)
        return to_bgra(img)

    def on_mouse(self, event, x, y, flags, param):
        cx, cy = self.mapper.to_canvas(x, y)
        self.panels.handle_mouse(event, cx, cy)

    # =====================================================
    # ВЫХОДНОЕ ВИДЕО
    # =====================================================

    def setup_video_writer(self):
        if not self.is_video_file:
            return

        name = os.path.splitext(os.path.basename(self.cap.filepath))[0]
        self.output_video_path = os.path.join(self.session_dir, name + "_analyzed.mp4")

        fps = self.cap.get_fps() or 30.0
        writer = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (CANVAS_W, CANVAS_H))
        if not writer.isOpened():
            print("Ошибка: не удалось создать выходное видео.")
            return
        self.video_writer = writer
        print("Выходное видео (без звука):", self.output_video_path)

    # =====================================================
    # ЗАЩИТА ОКНА ОТ ЗАХВАТА (режим экрана)
    # =====================================================

    def try_protect_window(self):
        if not self.is_screen_mode or self.window_protected or self.protect_attempts >= 30:
            return
        self.protect_attempts += 1
        if exclude_window_from_capture(self.WINDOW_TITLE):
            self.window_protected = True
            print("Окно предпросмотра исключено из захвата экрана.")
        elif self.protect_attempts == 30:
            print("Не удалось исключить окно предпросмотра из захвата экрана.")

    # =====================================================
    # ГЛАВНЫЙ ЦИКЛ
    # =====================================================

    def run(self):
        self.setup_video_writer()

        if not self.is_video_file and self.speech is not None:
            try:
                self.audio.start()
            except Exception as error:
                print("Не удалось открыть микрофон:", error)
        if self.speech is not None:
            self.speech.start()

        print()
        print("Видео, звук, субтитры, лицо, взгляд, мимика, покраснение и отклонение от базы запущены.")
        print("Клавиши:  q / Esc - выход   c - калибровка мимики   r - калибровка покраснения")
        print("          b - личная база (необязательно)   p - вернуть панели на места")
        print("          h - скрыть/показать панели   f - полный экран")
        print("Разметка:  t - сейчас говорится правда   l - сейчас ложь   n - снять метку")
        print("Панели переносятся мышью, окно можно растягивать.")
        print()

        cv2.namedWindow(self.WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_TITLE, CANVAS_W, CANVAS_H)
        cv2.setMouseCallback(self.WINDOW_TITLE, self.on_mouse)

        wall0 = time.time()
        p = self.pipeline

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Обработка видео завершена." if self.is_video_file
                      else "Ошибка чтения кадра.")
                break

            frame_time = self.timeline()
            self.video_time = frame_time
            self.feed_file_audio(frame_time)

            volume = self.audio.get_volume()
            lines = self.speech.get_display_lines() if self.speech is not None else []
            subtitle_text = lines[-1] if lines else ""
            if self.speech is not None:
                for t_speech, raw_tail in self.speech.drain_speech():
                    p.feed_speech(raw_tail, t_speech)

            # анализ кадра; рамка лица и контуры глаз рисуются прямо на кадре
            p.process(frame, frame_time, subtitle_text=subtitle_text, draw_overlays=True)
            p.drain_events()

            # --- панели ---
            if self.frame_index % PANEL_RENDER_PERIOD == 0:
                panels = self.panels
                panels.set_image("volume", self.render_volume(volume))
                panels.set_image("eyes", p.eye_tracker.render_panel(p.last_quality.note))
                panels.set_image("redness", p.redness_analyzer.render_panel(p.last_redness))
                panels.set_image("mimic", p.mimic_analyzer.render_panel(p.last_mimic, p.last_micro))
                panels.set_image("suspicion", p.suspicion_analyzer.render_panel(p.last_suspicion))
                panels.set_image("subtitles", self.render_subtitles(lines))
                panels.set_image("label", self.render_label())
            self.panels.draw(frame)

            if self.video_writer is not None:
                self.video_writer.write(frame)

            cv2.imshow(self.WINDOW_TITLE, self.mapper.render(frame))
            self.try_protect_window()
            self.frame_index += 1

            # --- реальное время для видеофайла ---
            if self.is_video_file and PACE_VIDEO_FILE:
                ahead = frame_time - (time.time() - wall0)
                if ahead > 0:
                    time.sleep(min(ahead, 0.1))

            # --- клавиатура ---
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                p.mimic_analyzer.start_calibration()
            elif key == ord("r"):
                p.redness_analyzer.start_calibration()
            elif key == ord("b"):
                p.suspicion_analyzer.start_calibration()
            elif key == ord("t"):
                self.set_label(LABEL_TRUTH)
            elif key == ord("l"):
                self.set_label(LABEL_LIE)
            elif key == ord("n"):
                self.set_label("")
            elif key == ord("p"):
                self.panels.reset_layout()
            elif key == ord("h"):
                self.panels.toggle_visible()
            elif key == ord("f"):
                self.fullscreen = not self.fullscreen
                cv2.setWindowProperty(
                    self.WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)

            # окно закрыто крестиком
            if cv2.getWindowProperty(self.WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break

        self.shutdown()

    def shutdown(self):
        if self.speech is not None:
            self.speech.stop(timeout=3.0)
        self.cap.release()
        self.audio.stop()
        self.pipeline.close()
        self.panels.save()

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            print("Обработанное видео сохранено:", self.output_video_path)

        cv2.destroyAllWindows()
        print("Результаты сохранены в папке:", self.session_dir)


def run_classic():
    source = choose_video_source(width=CANVAS_W, height=CANVAS_H)
    if not source.isOpened():
        print("Не удалось открыть источник видео. Проверьте камеру, её индекс или путь к файлу.")
        return 1
    who = input("ID участника для оценки (Enter - пропустить): ").strip()
    App(source, participant=who).run()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="All the girls — наблюдение за мимикой, взглядом и речью собеседника")
    parser.add_argument("--classic", action="store_true",
                        help="прежний интерфейс в окне OpenCV с выбором источника в консоли")
    args = parser.parse_args(argv)

    if not args.classic:
        try:
            from gui.app import run_gui
        except ImportError as error:
            print("Не удалось загрузить настольный интерфейс (PySide6):", error)
            print("Установите зависимости: pip install -r requirements-main.txt")
            print("Запускаю классический режим OpenCV...")
        else:
            return run_gui([sys.argv[0]])
    return run_classic()


if __name__ == "__main__":
    sys.exit(main())
