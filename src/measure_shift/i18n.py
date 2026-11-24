"""Presentation strings for the desktop UI. Stored values use the English keys."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re

from PySide6 import QtWidgets
from PySide6 import QtCore

from .german import GERMAN, WORKER_TEMPLATES


@dataclass(frozen=True)
class Term:
    """A translated argument inside a message. Other arguments are literal data."""

    source: str


@lru_cache(maxsize=256)
def german_detail(source):
    if source in GERMAN:
        return GERMAN[source]
    # Workers also expose their English messages to CLI callers. Localize their
    # formatted details at the GUI boundary, leaving unknown driver errors intact.
    for pattern, template in WORKER_TEMPLATES:
        match = re.fullmatch(pattern, source, re.DOTALL)
        if match:
            return template.format(**{key: german_detail(value) if key in ("reason", "direction", "label")
                                      else value for key, value in match.groupdict().items()})
    if "; " in source:
        return "; ".join(german_detail(part) for part in source.split("; "))
    return source


class _Binding(QtCore.QObject):
    """A text property follows language changes for the lifetime of its widget."""

    def __init__(self, language, setter):
        super().__init__(setter.__self__)
        self.language = language
        self.setter = setter
        self.message = None
        language.changed.connect(self.refresh)
        if setter.__name__ == "setToolTip" and isinstance(self.parent(), QtWidgets.QWidget):
            self.parent().installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.Type.ToolTip:
            QtWidgets.QToolTip.setFont(watched.font())
        return False

    @QtCore.Slot()
    def refresh(self):
        source, args = self.message
        self.setter(self.language.text(source, *args))


class Language(QtCore.QObject):
    changed = QtCore.Signal()

    def __init__(self, code="de", parent=None):
        super().__init__(parent)
        self.code = code

    def select(self, code):
        code = "en" if code == "en" else "de"
        if self.code != code:
            self.code = code
            self.changed.emit()

    def translate_standard_widgets(self):
        """Give Qt-owned controls, including file-dialog buttons, the app language."""
        app = QtWidgets.QApplication.instance()
        translator = getattr(app, "_german_qt_translator", None)
        if self.code == "de" and translator is None:
            translator = QtCore.QTranslator(app)
            directory = QtCore.QLibraryInfo.path(QtCore.QLibraryInfo.LibraryPath.TranslationsPath)
            if translator.load("qtbase_de", directory):
                app.installTranslator(translator)
                app._german_qt_translator = translator
            else:
                translator.deleteLater()
        elif self.code == "en" and translator is not None:
            app.removeTranslator(translator)
            app._german_qt_translator = None
            translator.deleteLater()

    def text(self, source, *args):
        template = german_detail(source) if self.code == "de" else source
        if not args:
            return template
        return template.format(*(self.text(arg.source) if isinstance(arg, Term) else arg for arg in args))

    def bind(self, setter, source, *args):
        target = setter.__self__
        if not hasattr(target, "_text_bindings"):
            target._text_bindings = {}
        binding = target._text_bindings.get(setter.__name__)
        if binding is None:
            binding = target._text_bindings[setter.__name__] = _Binding(self, setter)
        if binding.message != (source, args):
            binding.message = source, args
            binding.refresh()
        return target

    def label(self, text):
        return self.bind(QtWidgets.QLabel().setText, text)

    def button(self, text):
        return self.bind(QtWidgets.QPushButton().setText, text)

    def check(self, text):
        return self.bind(QtWidgets.QCheckBox().setText, text)

    def group(self, text):
        return self.bind(QtWidgets.QGroupBox().setTitle, text)

    def action(self, menu, text, *args):
        return self.bind(menu.addAction(text, *args).setText, text)

    def menu(self, parent, text):
        return self.bind(parent.addMenu(text).setTitle, text)


class ChoiceBox(QtWidgets.QComboBox):
    """Localizable choices with stable values for workers and settings files."""

    def __init__(self, language, choices):
        super().__init__()
        self.language = language
        for key in choices:
            self.addItem(language.text(key), key)
        language.changed.connect(self.refresh)

    @QtCore.Slot()
    def refresh(self):
        with QtCore.QSignalBlocker(self):
            for index in range(self.count()):
                self.setItemText(index, self.language.text(self.itemData(index)))

    def select(self, key):
        index = self.findData(key)
        if index >= 0:
            self.setCurrentIndex(index)
