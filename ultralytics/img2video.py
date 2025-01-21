import cv2
import os,glob
import natsort 

if __name__ == "__main__":

	image_folder = '/media/gilbert11/DATA/mnt/Flat/images'#./images'
	video_name = 'flat_pet2.mp4'

	images = natsort.natsorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
	print("======> " , images[0])
	print(type(images),images[:10])
	frame = cv2.imread(os.path.join(image_folder, images[0]))
	fps = 60
	height, width, layers = frame.shape
	#cap = cv2.VideoCapture(video_path)
	#width =  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
	#height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
	video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), fps , (width,height))

	for image in images:
	    video.write(cv2.imread(os.path.join(image_folder, image)))

	cv2.destroyAllWindows()
	video.release()

