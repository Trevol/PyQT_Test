# import enamlx
# enamlx.install()

from enaml import imports as enaml_imports
from enaml.qt.qt_application import QtApplication
from cv2 import imread, cvtColor, COLOR_BGR2RGB

with enaml_imports():
    from main_view import Main

app = QtApplication()
image = cvtColor(imread("../../images//contacts/3contacts.jpg"), COLOR_BGR2RGB)
view = Main(image=image, initial_size=(600, 600))
view.show()
#view.maximize()
app.start()
