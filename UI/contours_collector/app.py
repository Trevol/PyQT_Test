import sys_excepthook_setup
import enamlx

enamlx.install(allow_def=True)
from enaml import imports as enaml_imports
from enaml.qt.qt_application import QtApplication

with enaml_imports():
    from main import Main

app = QtApplication()
view = Main(initial_size=(1200, 800))
view.show()
view.maximize()
app.start()
