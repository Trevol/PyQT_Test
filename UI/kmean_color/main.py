# import enamlx
# enamlx.install()

from enaml import imports as enaml_imports
from enaml.qt.qt_application import QtApplication
import cv2

with enaml_imports():
    from main_view import Main

app = QtApplication()

#frame = "D:/DiskE/Computer_Vision_Task/frames_6/f_2770_184666.67_184.67.jpg"
frame = "../../images//contacts/3contacts.jpg"
image = cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2GRAY)

view = Main(image=image, initial_size=(600, 600))
view.show()
#view.maximize()
app.start()
