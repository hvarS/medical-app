import time

from qtpy.QtCore import QThread, Signal
from qtpy.QtWidgets import QWidget

from widgets.loading_translucent_screen import LoadingTranslucentScreen


class LoadingThread(QThread):
    loadingSignal = Signal()

    def __init__(self, loading_screen: LoadingTranslucentScreen, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__loadingTranslucentScreen = loading_screen
        self.__loadingTranslucentScreen.setParentThread(self)
        self.started.connect(self.__loadingTranslucentScreen.start)
        self.finished.connect(self.__loadingTranslucentScreen.stop)
        self.started.connect(self.__loadingTranslucentScreen.makeParentDisabledDuringLoading)
        self.finished.connect(self.__loadingTranslucentScreen.makeParentDisabledDuringLoading)

    def run(self):
        time.sleep(5)