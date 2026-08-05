"""Fantasia Conductor — application entry point.

Boots the Qt application and shows the main window. The UI layer talks to
``fantasia_core`` for all logic; this module wires nothing but the app shell.
"""

from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Fantasia Conductor")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
