import gtk
from threading import threading
import gobject 

gtk.gdk.threads_init()


class DisplayImage():

	def __init__(self, title="OpenCV"):
		self.img = None
		self.img_gtk = None
		self.done = False
		self.thrd = None
		self.win = gtk.Window()
		self.win.set_title(title)
		self.win.connect("delete_event",self.leave_app)
		self.image_box = gtk.EventBox()
		self.win.add(self.image_box)

	