import sys_excepthook_setup
import enamlx
enamlx.install()

import enaml
from enaml.qt.qt_application import QtApplication




with enaml.imports():
    from main_view import Main

app = QtApplication()
main = Main()
main.show()
app.start()