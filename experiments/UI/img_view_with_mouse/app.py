import enamlx
enamlx.install()

import enaml
from enaml.qt.qt_application import QtApplication

with enaml.imports():
    from main_view import MainView

app = QtApplication()
view = MainView()
view.show()
app.start()