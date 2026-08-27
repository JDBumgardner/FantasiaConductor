"""Fantasia Conductor — application entry point.

Boots the Qt application and shows the main window. The UI layer talks to
``fantasia_core`` for all logic; this module wires nothing but the app shell.
"""

from __future__ import annotations

import sys


# How long a thread may hold the GIL before it has to offer it up. The audio
# callback is Python, so it cannot run at all while another thread holds the
# lock — and at CPython's 5ms default, ordinary UI work (repainting the
# timeline, opening a panel) starves it badly enough to drop blocks, which is
# heard as popping.
#
# Measured against a 93ms callback deadline with three CPU-bound Python threads
# competing: at 5ms, 72 of 80 blocks missed. At 0.5ms, none did, and the median
# wait fell from 186ms to 20ms. Handing the lock over more often costs a little
# throughput on pure-Python work, which is not where this app spends its time —
# numpy, the plugins and the model runtimes all release the GIL while they work.
GIL_SWITCH_INTERVAL = 0.0005


def main() -> int:
    sys.setswitchinterval(GIL_SWITCH_INTERVAL)

    from PySide6.QtWidgets import QApplication

    from ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Fantasia Conductor")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
