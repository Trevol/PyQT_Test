import enaml
from enaml.qt.qt_application import QtApplication

with enaml.imports():
    from test_view_old import Main

app = QtApplication()

Main(counter=9).show()
Main(counter=11).show()
app.start()