from PySide2.QtCore import QObject, QEvent, Qt

class KeyInputFilter(QObject):
    def __init__(self, func, parent = None):
        super().__init__(parent)
        self.func = func
    
    def eventFilter(self, watched, event):
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Delete:
                self.func()
                return True
        return super().eventFilter(watched, event)