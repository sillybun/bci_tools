APP_STYLE = """
QWidget { color: #173f46; font-size: 13px; letter-spacing: 0px; }
QMainWindow, QDialog, QWidget#shell { background: #eaf3f2; }
QToolBar { background: #eaf3f2; border: 0; spacing: 7px; padding: 9px 14px 3px; }
QToolButton, QPushButton { min-height: 30px; border: 1px solid #c4dada; border-radius: 8px; padding: 0 10px; background: #f7fbfa; color: #17444c; }
QToolButton:hover, QPushButton:hover { background: #e1f0ee; border-color: #7fb8b2; }
QToolButton:pressed, QPushButton:pressed { background: #cde5e1; border-color: #2f7f7a; }
QToolButton:disabled, QPushButton:disabled { color: #91a9aa; background: #edf5f4; }
QPushButton#primary { color: #fff; background: #2f7f7a; border-color: #2f7f7a; }
QPushButton#danger { color: #b33b3b; border-color: #e4c4c4; }
QFrame#leftPane, QWidget#repoPage, QWidget#commitPage { background: #fff; border: 1px solid #cfe1e0; border-radius: 10px; }
QFrame#activityCard, QFrame#repoSettingsPanel { background: #f2f8f7; border: 1px solid #d6e7e5; border-radius: 10px; }
QToolButton#repoSettingsToggle { background: transparent; border-color: transparent; padding-left: 0; color: #1e6f70; }
QToolButton#repoSettingsToggle:hover { background: #e8f3f1; padding-left: 8px; }
QPushButton#commitCard { background: #fff; border: 1px solid #d4e4e3; border-radius: 10px; padding: 0; }
QPushButton#commitCard:hover { background: #f7fbfa; border-color: #8bbfba; }
QPushButton#commitCard:pressed { background: #eaf5f3; }
QPushButton#commitCard[selected="true"] { background: #eaf3f2; border: 2px solid #2f7f7a; }
QPushButton#commitCard[selected="true"]:hover { background: #e3f0ef; border-color: #206c68; }
QLabel#commitVersion { color: #176b73; background: #dceff2; border-radius: 7px; padding: 2px 6px; font-size: 11px; font-weight: 700; }
QLabel#commitRefs, QLabel#commitState, QLabel#commitTime { color: #648083; font-size: 11px; }
QLabel#commitSubject { color: #173f46; font-weight: 600; }
QLabel#sectionTitle { font-size: 15px; font-weight: 700; color: #173f46; }
QLabel#muted { color: #648083; }
QLabel#repoTitle { font-size: 23px; font-weight: 700; color: #123e47; }
QLabel#badge { border-radius: 8px; padding: 3px 7px; font-size: 11px; font-weight: 600; }
QListWidget { background: #fff; border: 0; outline: 0; padding: 3px; }
QListWidget::item { border-radius: 8px; padding: 9px 8px; margin: 1px 3px; }
QListWidget::item:selected { background: transparent; }
QListWidget#credentialHosts::item:selected { color: #173f46; background: #dcefe9; }
QLabel#loginIdentityTag { color: #176b63; background: #dcefe9; border: 1px solid #c1dfd8; border-radius: 8px; padding: 4px 8px; font-size: 11px; font-weight: 600; }
QScrollArea#commitScroll, QScrollArea#commitScroll QWidget#qt_scrollarea_viewport, QWidget#commitContainer { background: #fff; }
QScrollArea, QScrollArea#detailPanel, QScrollArea#detailPanel QWidget#qt_scrollarea_viewport { border: 0; background: transparent; border-radius: 8px; }
QTextBrowser#readme { background: transparent; border: 0; border-radius: 8px; padding: 14px; }
QTextBrowser#readme QScrollBar:vertical { background: transparent; width: 8px; margin: 2px 2px 2px 0; }
QTextBrowser#readme QScrollBar::handle:vertical { background: #9fc7c8; min-height: 28px; border-radius: 4px; }
QTextBrowser#readme QScrollBar::handle:vertical:hover { background: #5e9e9d; }
QTextBrowser#readme QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical, QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; height: 0; }
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QTableWidget { background: #eff6f5; border: 1px solid #c9dfdd; border-radius: 8px; padding: 7px; selection-background-color: #2f7f7a; }
QComboBox { padding-right: 34px; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 30px; border: 0; border-radius: 7px; background: #dceceb; }
QComboBox::drop-down:hover { background: #cbe3e0; }
QComboBox::down-arrow { width: 0; height: 0; }
QComboBox#branchSelector { padding-right: 7px; }
QComboBox#branchSelector::drop-down { width: 0; background: transparent; }
QToolButton#branchDropButton, QToolButton#cloneBranchDropButton { min-height: 0; padding: 0; border-radius: 7px; background: #dceceb; }
QToolButton#branchDropButton:hover, QToolButton#cloneBranchDropButton:hover { background: #cbe3e0; }
QComboBox QAbstractItemView { color: #173f46; background: #fff; border: 1px solid #c9dfdd; border-radius: 8px; padding: 5px; selection-color: #155a5b; selection-background-color: #dcefe9; outline: 0; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 2px 2px 2px 0; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal { background: #a7c9ca; border-radius: 4px; }
QScrollBar::handle:vertical { min-height: 28px; }
QScrollBar::handle:horizontal { min-width: 28px; }
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover { background: #6ca4a4; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical, QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; height: 0; }
QScrollBar:horizontal { background: transparent; height: 8px; margin: 0 2px 2px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal, QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; width: 0; }
QTableWidget { gridline-color: transparent; }
QHeaderView::section { background: #e2f0ef; border: 0; padding: 7px; font-weight: 600; color: #245b62; }
QTableCornerButton::section { background: #e2f0ef; border: 0; }
QProgressBar#diffProgress { border: 0; background: #e2f0ef; border-radius: 2px; min-height: 4px; max-height: 4px; }
QProgressBar#diffProgress::chunk { background: #2f7f7a; border-radius: 2px; }
QSplitter::handle { background: transparent; width: 12px; }
QStatusBar { background: #eaf3f2; border: 0; color: #527579; }
QMenu { background: #fff; border: 1px solid #cfe1e0; padding: 6px; }
QMenu::item { padding: 7px 24px 7px 9px; border-radius: 8px; }
QMenu::item:selected { background: #dcefe9; }
QMenu::separator { height: 1px; background: #dce8e7; margin: 5px 8px; }
"""

STATE_COLORS = {
    "未提交": ("#dcecf4", "#176079"), "同步": ("#d9f0e8", "#176b63"),
    "过期": ("#f9e4e1", "#a63c3c"), "断线": ("#e7eeee", "#587174"),
    "已提交": ("#dcecf4", "#176079"), "已抓取": ("#d9f0e8", "#176b63"),
    "已同步": ("#d9f0e8", "#176b63"), "未抓取": ("#f9e4e1", "#a63c3c"),
    "未推送": ("#e3edf4", "#3a6170"),
}
