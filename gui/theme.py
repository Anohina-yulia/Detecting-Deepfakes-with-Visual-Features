# -*- coding: utf-8 -*-
"""Дизайн-система: цвета, шрифты, стили.

Палитра намеренно узкая: белый, графит, мягкие серые и один красный акцент.
Красный означает только "активно" (идёт анализ, запись) или "обратите внимание"
(показатель вышел за свой обычный диапазон). Других цветов в интерфейсе нет.
"""
import os

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication


class C:
    BG = "#F2F2F4"
    SURFACE = "#FFFFFF"
    SURFACE_2 = "#F7F7F9"
    SURFACE_3 = "#EDEDF0"
    BORDER = "#E4E4E9"
    BORDER_2 = "#D3D3DA"
    TEXT = "#17171C"
    TEXT_2 = "#4A4A54"
    TEXT_3 = "#84848F"
    TEXT_4 = "#B4B4BD"

    STAGE = "#121316"           # фон видео
    GRAPHITE = "#1B1C21"
    GRAPHITE_2 = "#24252B"
    GRAPHITE_3 = "#2F3037"
    ON_DARK = "#F1F1F4"
    ON_DARK_2 = "#A4A5AF"
    ON_DARK_3 = "#6E6F79"

    ACCENT = "#E1323C"
    ACCENT_HOVER = "#C92B34"
    ACCENT_PRESSED = "#AE232B"
    ACCENT_SOFT = "#FDEDEE"
    ACCENT_SOFT_2 = "#F7D3D6"
    ACCENT_DISABLED = "#F0B7BC"


def qc(hex_color, alpha=None):
    color = QColor(hex_color)
    if alpha is not None:
        color.setAlpha(int(alpha))
    return color


class Fonts:
    ui = "Segoe UI"
    display = "Segoe UI"
    numeric = "Segoe UI"


def _pick(candidates, available):
    for name in candidates:
        if name in available:
            return name
    return QApplication.font().family()


def init_fonts():
    available = set(QFontDatabase.families())
    Fonts.ui = _pick(["Segoe UI Variable Text", "Segoe UI", "Arial"], available)
    Fonts.display = _pick(["Segoe UI Variable Display", "Segoe UI Variable Text", "Segoe UI",
                           "Arial"], available)
    # Bahnschrift - технический шрифт Windows с кириллицей: цифры как на приборе
    Fonts.numeric = _pick(["Bahnschrift", "Segoe UI Variable Display", "Segoe UI"], available)


_WEIGHTS = {300: QFont.Light, 400: QFont.Normal, 500: QFont.Medium,
            600: QFont.DemiBold, 700: QFont.Bold}


def font(px, weight=400, family=None, letter_spacing=None):
    f = QFont(family or Fonts.ui)
    f.setPixelSize(int(round(px)))
    if isinstance(weight, int):
        weight = _WEIGHTS.get(weight, QFont.Normal)
    f.setWeight(weight)
    f.setHintingPreference(QFont.PreferNoHinting)
    if letter_spacing is not None:
        f.setLetterSpacing(QFont.AbsoluteSpacing, letter_spacing)
    return f


def display_font(px, weight=600):
    return font(px, weight, Fonts.display)


def numeric_font(px, weight=400):
    return font(px, weight, Fonts.numeric)


ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _asset(name):
    return os.path.join(ASSETS_DIR, name).replace("\\", "/")


def stylesheet():
    return f"""
* {{ outline: none; }}
/* шрифт задаётся через QApplication.setFont: правило font-* здесь перебило бы setFont виджетов */
QWidget {{ color: {C.TEXT}; }}
QMainWindow, QWidget#Central, QDialog {{ background: {C.BG}; }}
QToolTip {{
    background: {C.GRAPHITE};
    color: {C.ON_DARK};
    border: 1px solid {C.GRAPHITE_3};
    padding: 8px 10px;
    font-size: 12px;
}}

/* ---------- верхняя и нижняя панели ---------- */
QFrame#TopBar {{ background: {C.SURFACE}; border-bottom: 1px solid {C.BORDER}; }}
QFrame#ControlBar {{ background: {C.SURFACE}; border-top: 1px solid {C.BORDER}; }}
QFrame#VSep {{ background: {C.BORDER}; min-width: 1px; max-width: 1px; }}
QLabel#AppName {{ font-family: "{Fonts.display}"; font-size: 16px; font-weight: 600; color: {C.TEXT}; }}
QLabel#Muted {{ color: {C.TEXT_3}; font-size: 12px; }}
QLabel#Caption {{ color: {C.TEXT_3}; font-size: 11px; font-weight: 600; }}
QLabel#Clock {{ font-family: "{Fonts.numeric}"; font-size: 16px; color: {C.TEXT}; }}
QLabel#CardTitle {{ font-family: "{Fonts.display}"; font-size: 15px; font-weight: 600; color: {C.TEXT}; }}
QLabel#CardSubtitle {{ color: {C.TEXT_3}; font-size: 12px; }}
QLabel#Hint {{ color: {C.TEXT_3}; font-size: 12px; }}
QLabel#SectionTitle {{ font-family: "{Fonts.display}"; font-size: 14px; font-weight: 600; }}
QLabel#FieldTitle {{ font-size: 13px; color: {C.TEXT}; }}
QLabel#FieldHint {{ font-size: 12px; color: {C.TEXT_3}; }}
QLabel#DialogTitle {{ font-family: "{Fonts.display}"; font-size: 20px; font-weight: 600; }}
QLabel#LiveText {{ color: {C.TEXT_3}; font-style: italic; font-size: 12px; }}

/* ---------- кнопки ---------- */
QPushButton {{
    background: {C.SURFACE};
    border: 1px solid {C.BORDER_2};
    border-radius: 10px;
    padding: 0px 14px;
    min-height: 38px;
    color: {C.TEXT};
    font-weight: 500;
}}
QPushButton:hover {{ background: {C.SURFACE_2}; border-color: #C3C3CC; }}
QPushButton:pressed {{ background: {C.SURFACE_3}; }}
QPushButton:disabled {{ color: {C.TEXT_4}; border-color: {C.BORDER}; background: {C.SURFACE}; }}

QPushButton[variant="primary"] {{
    background: {C.ACCENT}; border: 1px solid {C.ACCENT}; color: #FFFFFF; font-weight: 600;
}}
QPushButton[variant="primary"]:hover {{ background: {C.ACCENT_HOVER}; border-color: {C.ACCENT_HOVER}; }}
QPushButton[variant="primary"]:pressed {{ background: {C.ACCENT_PRESSED}; border-color: {C.ACCENT_PRESSED}; }}
QPushButton[variant="primary"]:disabled {{
    background: {C.ACCENT_DISABLED}; border-color: {C.ACCENT_DISABLED}; color: #FFFFFF;
}}
QPushButton[variant="dark"] {{
    background: {C.GRAPHITE}; border: 1px solid {C.GRAPHITE}; color: #FFFFFF; font-weight: 600;
}}
QPushButton[variant="dark"]:hover {{ background: {C.GRAPHITE_3}; border-color: {C.GRAPHITE_3}; }}
QPushButton[variant="dark"]:pressed {{ background: #000000; }}
QPushButton[variant="ghost"] {{ background: transparent; border: 1px solid transparent; color: {C.TEXT_2}; }}
QPushButton[variant="ghost"]:hover {{ background: {C.SURFACE_3}; color: {C.TEXT}; }}
QPushButton[variant="ghost"]:disabled {{ color: {C.TEXT_4}; background: transparent; }}
QPushButton[variant="danger"] {{ color: {C.ACCENT}; border-color: {C.ACCENT_SOFT_2}; }}
QPushButton[variant="danger"]:hover {{ background: {C.ACCENT_SOFT}; border-color: {C.ACCENT_SOFT_2}; }}
QPushButton[variant="danger"]:disabled {{ color: {C.TEXT_4}; border-color: {C.BORDER}; }}
QPushButton[variant="onDark"] {{
    background: {C.GRAPHITE_2}; border: 1px solid {C.GRAPHITE_3}; color: {C.ON_DARK};
}}
QPushButton[variant="onDark"]:hover {{ background: {C.GRAPHITE_3}; }}
QPushButton[size="lg"] {{ min-height: 44px; font-size: 14px; padding: 0px 22px; border-radius: 12px; }}
QPushButton[size="sm"] {{ min-height: 30px; font-size: 12px; padding: 0px 10px; border-radius: 8px; }}
QPushButton[iconOnly="true"] {{ padding: 0px; min-width: 38px; max-width: 38px; }}

/* ---------- поля ввода ---------- */
QLineEdit, QComboBox, QSpinBox {{
    background: {C.SURFACE};
    border: 1px solid {C.BORDER_2};
    border-radius: 9px;
    padding: 0px 10px;
    min-height: 34px;
    selection-background-color: {C.ACCENT_SOFT_2};
    selection-color: {C.TEXT};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{ border-color: {C.TEXT_3}; }}
QLineEdit:disabled, QComboBox:disabled {{ color: {C.TEXT_4}; background: {C.SURFACE_2}; }}
QComboBox::drop-down {{ border: none; width: 28px; }}
QComboBox::down-arrow {{ image: url("{_asset('chevron-down.svg')}"); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{
    background: {C.SURFACE};
    border: 1px solid {C.BORDER};
    padding: 4px;
    selection-background-color: {C.SURFACE_3};
    selection-color: {C.TEXT};
    outline: none;
}}

/* ---------- списки, таблицы, текст ---------- */
QListWidget, QTreeWidget, QTextBrowser, QPlainTextEdit {{
    background: {C.SURFACE};
    border: 1px solid {C.BORDER};
    border-radius: 12px;
}}
QTextBrowser#Bare, QTextBrowser#Bare:focus {{ border: none; background: transparent; }}
QListWidget::item {{ border: none; padding: 0px; }}
QListWidget::item:selected {{ background: {C.SURFACE_3}; color: {C.TEXT}; border-radius: 10px; }}
QListWidget::item:hover {{ background: {C.SURFACE_2}; border-radius: 10px; }}
QTreeWidget::item {{ min-height: 30px; }}
QTreeWidget::item:selected {{ background: {C.SURFACE_3}; color: {C.TEXT}; }}
QHeaderView::section {{
    background: {C.SURFACE};
    color: {C.TEXT_3};
    border: none;
    border-bottom: 1px solid {C.BORDER};
    padding: 6px 8px;
    font-size: 11px;
    font-weight: 600;
}}

/* ---------- вкладки ---------- */
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{
    background: transparent;
    color: {C.TEXT_3};
    padding: 8px 2px;
    margin-right: 18px;
    border-bottom: 2px solid transparent;
    font-weight: 600;
}}
QTabBar::tab:selected {{ color: {C.TEXT}; border-bottom: 2px solid {C.ACCENT}; }}
QTabBar::tab:hover {{ color: {C.TEXT}; }}

/* ---------- прокрутка ---------- */
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #D2D2D9; border-radius: 3px; min-height: 28px; margin: 0px 2px; }}
QScrollBar::handle:vertical:hover {{ background: #BBBBC4; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #D2D2D9; border-radius: 3px; min-width: 28px; margin: 2px 0px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0px; height: 0px; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QMessageBox {{ background: {C.SURFACE}; }}
QMessageBox QLabel {{ color: {C.TEXT}; }}
QMenu {{ background: {C.SURFACE}; border: 1px solid {C.BORDER}; padding: 6px; }}
QMenu::item {{ padding: 7px 14px; border-radius: 6px; }}
QMenu::item:selected {{ background: {C.SURFACE_3}; }}
"""


def apply_theme(app):
    init_fonts()
    app.setFont(font(13))
    palette = app.palette()
    palette.setColor(QPalette.Window, qc(C.BG))
    palette.setColor(QPalette.Base, qc(C.SURFACE))
    palette.setColor(QPalette.AlternateBase, qc(C.SURFACE_2))
    palette.setColor(QPalette.Text, qc(C.TEXT))
    palette.setColor(QPalette.WindowText, qc(C.TEXT))
    palette.setColor(QPalette.ButtonText, qc(C.TEXT))
    palette.setColor(QPalette.Button, qc(C.SURFACE))
    palette.setColor(QPalette.Highlight, qc(C.ACCENT_SOFT_2))
    palette.setColor(QPalette.HighlightedText, qc(C.TEXT))
    palette.setColor(QPalette.ToolTipBase, qc(C.GRAPHITE))
    palette.setColor(QPalette.ToolTipText, qc(C.ON_DARK))
    palette.setColor(QPalette.PlaceholderText, qc(C.TEXT_4))
    app.setPalette(palette)
    app.setStyleSheet(stylesheet())


def repolish(widget):
    """Применить стиль после смены динамического свойства (variant и т.п.)."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()
