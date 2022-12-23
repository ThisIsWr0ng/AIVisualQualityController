import gi
import cv2
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk as gtk

class Main:
    def __init__(self):
        gladeFile = "Main.glade"
        self.builder = gtk.Builder()
        self.builder.add_from_file(gladeFile)
        self.builder.connect_signals(self)

        window = self.builder.get_object("Main")
        window.connect("delete-event", gtk.main_quit)
        window.show()

    def startPreview(self, widget):
        url = "https://192.168.1.239:8080/video"# link to smartphone camera
        webcam = 0 # 0 for laptop webcam / 1 for external webcam
        vc = cv2.VideoCapture(url)
        previewWindow = self.builder.get_object("previewWindow")

        if vc.isOpened(): # try to get the first frame
            rval, frame = vc.read()
        else:
            vc = cv2.VideoCapture(webcam)
            rval, frame = vc.read()
            #rval = False

        while rval:
            cv2.imshow("Main", frame)
            rval, frame = vc.read()
            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break

        vc.release()
        cv2.destroyWindow("Main")


if __name__ == '__main__':
    main = Main()
    gtk.main()