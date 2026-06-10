"""Design tokens + application stylesheet for the VESP-UQ Mission Console.

One dark, low-glare palette in the idiom of modern operations dashboards: a near-black canvas,
elevated card surfaces with 1px hairline borders and rounded corners, a single cool accent for
primary actions/selection, and semantic colors reserved for state (ok / warn / danger). All
colors live in :data:`TOKENS` so widgets and the QSS never hardcode hex values twice.
"""

from __future__ import annotations

TOKENS = {
    # canvas / surfaces
    "bg": "#0c1016",
    "surface": "#11161e",
    "card": "#161d27",
    "card_hover": "#1a2230",
    "field": "#0e131b",
    "border": "#232d3b",
    "border_soft": "#1c2532",
    # text
    "text": "#e7edf4",
    "text_muted": "#93a1b3",
    "text_faint": "#5d6b7d",
    # accent + semantic
    "accent": "#4da3ff",
    "accent_soft": "#15263c",
    "ok": "#46d39a",
    "ok_soft": "#11302a",
    "warn": "#ffc266",
    "warn_soft": "#33270f",
    "danger": "#ff6b6b",
    "danger_soft": "#361519",
    # console
    "console_bg": "#0a0e14",
    "console_text": "#b9c6d4",
    "mono": "Cascadia Code, Consolas, monospace",
    "radius": "10px",
}


def build_qss() -> str:
    """The full application stylesheet (interpolated from :data:`TOKENS`)."""

    t = TOKENS
    return f"""
* {{
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 10pt;
    color: {t['text']};
}}
QMainWindow, QWidget {{ background: {t['bg']}; }}
QToolTip {{
    background: {t['card']}; color: {t['text']};
    border: 1px solid {t['border']}; padding: 6px 8px;
}}

/* ---------------- navigation rail ---------------- */
#NavRail {{
    background: {t['surface']};
    border-right: 1px solid {t['border_soft']};
}}
#NavBrand {{
    font-size: 13pt; font-weight: 700; letter-spacing: 1px;
    color: {t['text']}; padding: 18px 16px 2px 16px;
}}
#NavBrandSub {{
    font-size: 8pt; color: {t['text_faint']};
    padding: 0 16px 14px 17px; letter-spacing: 2px;
}}
QPushButton[nav="true"] {{
    background: transparent; border: none; border-radius: 8px;
    text-align: left; padding: 10px 14px; margin: 2px 10px;
    color: {t['text_muted']}; font-weight: 600;
}}
QPushButton[nav="true"]:hover {{ background: {t['card_hover']}; color: {t['text']}; }}
QPushButton[nav="true"]:checked {{
    background: {t['accent_soft']}; color: {t['accent']};
}}
#NavFooter {{ color: {t['text_faint']}; font-size: 8pt; padding: 10px 16px; }}

/* ---------------- page chrome ---------------- */
#PageTitle {{ font-size: 17pt; font-weight: 700; }}
#PageSubtitle {{ color: {t['text_muted']}; font-size: 10pt; }}
#SectionTitle {{
    color: {t['text_muted']}; font-size: 8.5pt; font-weight: 700;
    letter-spacing: 1.6px; margin-top: 2px;
}}

/* ---------------- cards & tiles ---------------- */
QFrame[card="true"] {{
    background: {t['card']};
    border: 1px solid {t['border_soft']};
    border-radius: {t['radius']};
}}
QFrame[card="true"]:hover {{ border-color: {t['border']}; }}
#KpiValue {{ font-size: 16pt; font-weight: 700; }}
#KpiLabel {{ color: {t['text_muted']}; font-size: 8.5pt; letter-spacing: 1px; font-weight: 600; }}
#KpiHint  {{ color: {t['text_faint']}; font-size: 8pt; }}

/* ---------------- status chips ---------------- */
QLabel[chip="neutral"] {{
    background: {t['card_hover']}; color: {t['text_muted']};
    border: 1px solid {t['border']}; border-radius: 9px; padding: 3px 10px; font-weight: 600;
}}
QLabel[chip="ok"] {{
    background: {t['ok_soft']}; color: {t['ok']};
    border: 1px solid {t['ok_soft']}; border-radius: 9px; padding: 3px 10px; font-weight: 600;
}}
QLabel[chip="warn"] {{
    background: {t['warn_soft']}; color: {t['warn']};
    border: 1px solid {t['warn_soft']}; border-radius: 9px; padding: 3px 10px; font-weight: 600;
}}
QLabel[chip="danger"] {{
    background: {t['danger_soft']}; color: {t['danger']};
    border: 1px solid {t['danger_soft']}; border-radius: 9px; padding: 3px 10px; font-weight: 600;
}}
QLabel[chip="accent"] {{
    background: {t['accent_soft']}; color: {t['accent']};
    border: 1px solid {t['accent_soft']}; border-radius: 9px; padding: 3px 10px; font-weight: 600;
}}

/* ---------------- buttons ---------------- */
QPushButton {{
    background: {t['card_hover']}; border: 1px solid {t['border']};
    border-radius: 8px; padding: 7px 16px; font-weight: 600;
}}
QPushButton:hover {{ background: #222c3b; }}
QPushButton:pressed {{ background: {t['card']}; }}
QPushButton:disabled {{ color: {t['text_faint']}; background: {t['card']}; }}
QPushButton[variant="primary"] {{
    background: {t['accent']}; color: #08111d; border: none;
}}
QPushButton[variant="primary"]:hover {{ background: #6cb4ff; }}
QPushButton[variant="primary"]:disabled {{ background: {t['accent_soft']}; color: {t['text_faint']}; }}
QPushButton[variant="danger"] {{
    background: transparent; color: {t['danger']}; border: 1px solid {t['danger']};
}}
QPushButton[variant="danger"]:hover {{ background: {t['danger_soft']}; }}
QPushButton[variant="ghost"] {{ background: transparent; border: 1px solid {t['border']}; }}
QPushButton[variant="ghost"]:hover {{ background: {t['card_hover']}; }}

/* ---------------- inputs ---------------- */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {t['field']}; border: 1px solid {t['border']};
    border-radius: 7px; padding: 6px 9px; selection-background-color: {t['accent_soft']};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {t['accent']};
}}
QLineEdit:disabled, QComboBox:disabled {{ color: {t['text_faint']}; }}
QComboBox::drop-down {{ border: none; width: 26px; }}
QComboBox::down-arrow {{
    image: none; border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid {t['text_muted']}; margin-right: 9px;
}}
QComboBox QAbstractItemView {{
    background: {t['card']}; border: 1px solid {t['border']};
    selection-background-color: {t['accent_soft']}; outline: none;
}}
QCheckBox, QRadioButton {{ spacing: 8px; color: {t['text']}; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 15px; height: 15px; border: 1px solid {t['border']};
    border-radius: 4px; background: {t['field']};
}}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {t['accent']}; border-color: {t['accent']};
}}

/* ---------------- tables ---------------- */
QTableWidget, QTableView {{
    background: {t['surface']}; border: 1px solid {t['border_soft']};
    border-radius: 8px; gridline-color: {t['border_soft']};
    alternate-background-color: {t['card']};
    selection-background-color: {t['accent_soft']}; selection-color: {t['text']};
}}
QHeaderView::section {{
    background: {t['card']}; color: {t['text_muted']};
    border: none; border-bottom: 1px solid {t['border']};
    padding: 7px 10px; font-weight: 700; font-size: 8.5pt; letter-spacing: 0.6px;
}}
QTableCornerButton::section {{ background: {t['card']}; border: none; }}

/* ---------------- console / text views ---------------- */
QPlainTextEdit, QTextBrowser {{
    background: {t['console_bg']}; color: {t['console_text']};
    border: 1px solid {t['border_soft']}; border-radius: 8px; padding: 8px;
}}
QPlainTextEdit {{ font-family: {t['mono']}; font-size: 9pt; }}

/* ---------------- tabs ---------------- */
QTabWidget::pane {{ border: 1px solid {t['border_soft']}; border-radius: 8px; top: -1px; }}
QTabBar::tab {{
    background: transparent; color: {t['text_muted']};
    padding: 8px 16px; border: none; font-weight: 600;
}}
QTabBar::tab:selected {{ color: {t['accent']}; border-bottom: 2px solid {t['accent']}; }}
QTabBar::tab:hover {{ color: {t['text']}; }}

/* ---------------- misc ---------------- */
QProgressBar {{
    background: {t['field']}; border: none; border-radius: 3px;
    height: 6px; text-align: center; color: transparent;
}}
QProgressBar::chunk {{ background: {t['accent']}; border-radius: 3px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {t['border']}; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {t['text_faint']}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {t['border']}; border-radius: 5px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QSplitter::handle {{ background: {t['border_soft']}; width: 1px; }}
QStatusBar {{ background: {t['surface']}; color: {t['text_muted']}; border-top: 1px solid {t['border_soft']}; }}
"""
