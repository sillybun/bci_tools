from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from integrated_github_repos.ui.main_window import MainWindow
from integrated_github_repos.ui.theme import APP_STYLE


def resource_path(relative: str) -> Path:
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
    return bundle_root / relative


def main() -> int:
    if os.environ.get("IGR_GIT_ASKPASS") == "1":
        from integrated_github_repos.credentials import CredentialStore

        QCoreApplication.setApplicationName("Integrated GitHub Repos")
        QCoreApplication.setOrganizationName("Bertie97")
        answer = CredentialStore().answer_prompt(sys.argv[1] if len(sys.argv) > 1 else "")
        if answer is None:
            return 1
        os.write(1, (answer + "\n").encode("utf-8"))
        return 0
    app = QApplication(sys.argv)
    app.setApplicationName("Integrated GitHub Repos")
    app.setOrganizationName("Bertie97")
    app.setWindowIcon(QIcon(str(resource_path("assets/app-icon.png"))))
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
