import pygame
from time import sleep
pygame.camera.init()
cams = pygame.camera.list_cameras()

cam = pygame.camera.Camera(cams[0] , (640,480) , "RGB")
cam.start()
int frame = 1
while True:
	k = cam.get_image()
	pygame.image.save(k,str(frame)+".png")
	frame = int(frame) + 1
	sleep(60)