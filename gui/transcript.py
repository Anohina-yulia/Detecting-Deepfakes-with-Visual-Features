# -*- coding: utf-8 -*-
"""Карточка «Речь и субтитры»: последние распознанные фразы и живая строка."""
import html

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QTextBrowser

from gui.theme import C, Fonts
from gui.widgets import Card, CardHeader, Chip, label, make_button

MAX_SHOWN = 300


def fmt_time(t):
    t = max(0, int(t))
    return f"{t // 60:02d}:{t % 60:02d}" if t < 3600 else f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"


class TranscriptCard(Card):
    open_journal = Signal()

    def __init__(self, parent=None):
        super().__init__(parent, margins=(18, 14, 18, 12))
        self.body.setSpacing(6)
        header = CardHeader("Речь и субтитры")
        self.status = Chip("выключено", "mic")
        self.status.set("выключено", "muted")
        header.add_right(self.status)
        self.body.addWidget(header)

        self.view = QTextBrowser()
        self.view.setObjectName("Bare")
        # забирает оставшееся место, но не раздувает колонку (иначе появляется прокрутка)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        self.view.setMinimumHeight(56)
        self.view.setOpenLinks(False)
        self.view.setFrameShape(QTextBrowser.NoFrame)
        self.view.document().setDocumentMargin(2)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.body.addWidget(self.view, 1)

        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        self.live = label("", "LiveText")
        self.live.setWordWrap(True)
        bottom.addWidget(self.live, 1)
        self.journal_button = make_button("Вся история", "journal", "ghost", "sm",
                                          tooltip="Субтитры, события и системный журнал сессии")
        self.journal_button.clicked.connect(self.open_journal.emit)
        bottom.addWidget(self.journal_button, 0, Qt.AlignBottom)
        self.body.addLayout(bottom)

        self.phrases = []
        self.clear()

    def clear(self):
        self.phrases = []
        self.view.setHtml(self._placeholder())
        self.live.setText("")

    def _placeholder(self):
        return (f"<div style='color:{C.TEXT_4}; font-size:12px; margin-top:6px'>"
                "Здесь появятся распознанные фразы, когда собеседник начнёт говорить.</div>")

    def add_phrase(self, t, text):
        self.phrases.append((t, text))
        if len(self.phrases) > MAX_SHOWN:
            self.phrases = self.phrases[-MAX_SHOWN:]
            self._render_all()
            return
        if len(self.phrases) == 1:
            self.view.clear()
        self.view.append(self._phrase_html(t, text))
        bar = self.view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _render_all(self):
        self.view.setHtml("".join(self._phrase_html(t, x) for t, x in self.phrases))
        bar = self.view.verticalScrollBar()
        bar.setValue(bar.maximum())

    @staticmethod
    def _phrase_html(t, text):
        return (f"<p style='margin:0 0 7px 0'>"
                f"<span style='font-family:\"{Fonts.numeric}\"; color:{C.TEXT_3}; font-size:12px'>"
                f"{fmt_time(t)}</span>&nbsp;&nbsp;"
                f"<span style='color:{C.TEXT}; font-size:13px'>{html.escape(text)}</span></p>")

    def set_live(self, text):
        text = (text or "").strip()
        if self.phrases and text and text == self.phrases[-1][1]:
            text = ""
        self.live.setText(f"… {text}" if text else "")

    def set_status(self, text, tone="neutral"):
        self.status.set(text, tone, "mic")
