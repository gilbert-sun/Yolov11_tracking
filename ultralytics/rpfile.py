import glob,os
from pathlib import Path
 
#f1 = open("/home/gilbert11/Downloads/ultralytics/ultralytics/cfg/flat_v1/labels/test/*.txt", "r")
#f2 = open("/home/gilbert11/Downloads/ultralytics/ultralytics/cfg/flat_v1/labels/val/*.txt", "r")
#f3 = open("/home/gilbert11/Downloads/ultralytics/ultralytics/cfg/flat_v1/labels/train/*.txt", "r")
#w1 = open("/home/gilbert11/Downloads/ultralytics/ultralytics/cfg/pet_dataset_v1/labels/test/*.txt, "a")
#w2 = open("/home/gilbert11/Downloads/ultralytics/ultralytics/cfg/pet_dataset_v1/labels/val/*.txt", "a")
#w3 = open("/home/gilbert11/Downloads/ultralytics/ultralytics/cfg/pet_dataset_v1/labels/train/*.txt", "a")

#flist1 = glob.glob(f1)
#flist2 = glob.glob(f2)
#flist3 = glob.glob(f3)
#wlist1 = glob.glob(w1)
#wlist2 = glob.glob(w2)
#wlist3 = glob.glob(w3)



#for ff in flist1:
if __name__ == "__main__":

	flist1 = glob.glob("/home/gilbert11/Downloads/ultralytics/ultralytics/cfg/flat_v1/labels/test/*.txt")
	count = 0
	for aa in flist1:
		#aa = "/home/gilbert11/Downloads/ultralytics/ultralytics/cfg/flat_v1/labels/test/flat_54.txt"
		bb = open(aa,"r")
		context = (bb.readlines())

		out_name = Path(aa).parents[0] / Path('out') / Path(aa).name
		if not os.path.exists(Path(aa).parents[0] / Path('out')):
			os.makedirs(Path(aa).parents[0] / Path('out'))

		for id in context:
			w1 = open(out_name, "a+")
			count = len(id) 
			idd = id.strip("\n").split(" ")
			ide ="5 "+ idd[1]+" "+idd[2]+" "+ idd[3]+" "+idd[4]+"\n"
			#print("\n10-----------------> ",idd)
			#print("11-----------------> ",ide)
			w1.write(ide)
			w1.close() 
	print(count , " >1-----------------------------------> ")
		
	flist2 = glob.glob("/home/gilbert11/Downloads/ultralytics/ultralytics/cfg/flat_v1/labels/val/*.txt")
	count = 0
	for aa in flist2:
		#aa = "/home/gilbert11/Downloads/ultralytics/ultralytics/cfg/flat_v1/labels/test/flat_54.txt"
		bb = open(aa,"r")
		context = (bb.readlines())

		out_name = Path(aa).parents[0] / Path('out') / Path(aa).name
		if not os.path.exists(Path(aa).parents[0] / Path('out')):
			os.makedirs(Path(aa).parents[0] / Path('out'))

		for id in context:
			w1 = open(out_name, "a+")
			count = len(id) 
			idd = id.strip("\n").split(" ")
			ide ="5 "+ idd[1]+" "+idd[2]+" "+ idd[3]+" "+idd[4]+"\n"
			#print("\n20-----------------> ",idd)
			#print("21-----------------> ",ide)
			w1.write(ide)
			w1.close() 
	print(count , " >2-----------------------------------> ")

	flist3 = glob.glob("/home/gilbert11/Downloads/ultralytics/ultralytics/cfg/flat_v1/labels/train/*.txt")
	count = 0
	for aa in flist3:
		#aa = "/home/gilbert11/Downloads/ultralytics/ultralytics/cfg/flat_v1/labels/test/flat_54.txt"
		bb = open(aa,"r")
		context = (bb.readlines())

		out_name = Path(aa).parents[0] / Path('out') / Path(aa).name
		if not os.path.exists(Path(aa).parents[0] / Path('out')):
			os.makedirs(Path(aa).parents[0] / Path('out'))

		for id in context:
			w1 = open(out_name, "a+")
			count = len(id) 
			idd = id.strip("\n").split(" ")
			ide ="5 "+ idd[1]+" "+idd[2]+" "+ idd[3]+" "+idd[4]+"\n"
			#print("\n30-----------------> ",idd)
			#print("31-----------------> ",ide)
			w1.write(ide)
			w1.close() 
	print(count , " >3-----------------------------------> ")	
	os.system("cp /home/gilbert11/Downloads/ultralytics/ultralytics/cfg/flat_v1/labels/test/out/* /home/gilbert11/Downloads/ultralytics/ultralytics/cfg/pet_dataset_v1/labels/test/")

	os.system("cp /home/gilbert11/Downloads/ultralytics/ultralytics/cfg/flat_v1/labels/val/out/* /home/gilbert11/Downloads/ultralytics/ultralytics/cfg/pet_dataset_v1/labels/val/")

	os.system("cp /home/gilbert11/Downloads/ultralytics/ultralytics/cfg/flat_v1/labels/train/out/* /home/gilbert11/Downloads/ultralytics/ultralytics/cfg/pet_dataset_v1/labels/train/")

	print( "End-----------------------------------> ")
 
