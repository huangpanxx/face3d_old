#coding:utf8
import os
import sys

def main():
	if len(sys.argv) <= 1:
		print u'Usage: drag a picture with human face onto this file'
		os.system('pause')
		exit(-1)

	image_dir = os.path.dirname(sys.argv[1])
	os.chdir(image_dir)

	curdir = os.path.dirname(os.path.abspath(__file__))
	image = sys.argv[1]
	buildcmd = '""%s/face3d.exe" "%s""' % (curdir,image)
	showcmd =  '""%s/showface.exe" "%s""' % (curdir,image)

	if os.path.exists( image_dir + '/F'):
		os.remove( image_dir + '/F')


	print buildcmd
	os.system(buildcmd) #Build 3d model


	if os.path.exists( image_dir + '/F'): #Build success?
		os.system(showcmd) #Show it!
		os.remove( image_dir + '/A')
		os.remove( image_dir + '/B')
		os.remove( image_dir + '/F')
		os.remove( image_dir + '/BOUND')
		os.remove( image_dir + '/MAP')
		os.remove( image_dir + '/I.bmp')
if __name__ == '__main__':
	try:
		main()
		os.system("pause")
	except Exception,e:
		print u'错误:%s' % e
		os.system("pause")
