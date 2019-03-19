from enaml import imports as enaml_imports
from enaml.qt.qt_application import QtApplication

with enaml_imports():
    from play_video_main_view import Main

app = QtApplication()
view = Main(initial_size=(1200, 800))
view.show()
#view.maximize()
app.start()
