import sys
import numpy as np
import cv2

def extractImg(name,numImgs):
	#Load the images
    imgList=[];
    for imgIdx in range(numImgs):
		filename = name + '_' + str(imgIdx+1) + '.png';
		img = cv2.imread(filename)
		if img is None:
			print "Error. Cannot read image";
			print filename;
			sys.exit();
		np.set_printoptions(threshold=np.nan)

		# convert to grayscale
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray,(5,5),0)
		thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,
									15,2)
		#_,thresh = cv2.threshold(gray,65,255,cv2.THRESH_BINARY_INV)

		#kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
		#dilated = cv2.dilate(thresh,kernel,iterations = 13) # dilate
		fthresh=np.float32(thresh)
		fthresh = fthresh/255
		#Split the images to lines.
		
		imLinesTuple = getImageLinesTuple(fthresh, filename);
		print " imLinesTuple";
		print len(imLinesTuple);
		imgList = []	
		#Find the contours in the given image
		for imtup in imLinesTuple:
			cntImg = imtup[0]*255;
			cntImg = np.uint8(cntImg);
			cntImgCp = np.array(cntImg);
			_,contours,hierarchy = cv2.findContours(cntImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
			op = [];
			for cnt in contours:
				x,y,w,h = cv2.boundingRect(cnt)
				if w < 400 and h > 5 and w > 5:
					cv2.rectangle(cntImgCp,(x,y),(x+w,y+h),(0,255,0),2);
					item = cntImgCp[y:y+h, x:x+w];
					itemr = cv2.resize(item,(40,40));
					op.append((itemr,x));
			op.sort(key=lambda tup:tup[1]);
			print "op size ", len(op)
			print "imtup size ", len(imtup[1])
			for img,label in zip(op,imtup[1]):
				imgList.append((img[0],label));
               

def getImageLinesTuple(fthresh, filename):
	startidx=0
	endidx=0
	imlist = []
	labellinelist=[]
	histidx = 0;
	minthresh = 3;
	hist = np.sum(fthresh,axis=1)
	print "In getImageLinesTuple. len is ", len(hist);

	while( histidx < len(hist)):
		if hist[histidx] > minthresh:
			startidx = histidx;
			endidx = startidx;
			while(endidx  < len(hist)):
				if hist[endidx] <= minthresh:
					tmpimg = fthresh[startidx:endidx,]
					imlist.append(tmpimg)
					break;
				endidx = endidx+1;
			histidx = histidx+(endidx-startidx);
		else:
			histidx = histidx+1;
	print "In getImageLinesTuple";
	#Read the labels line
	labelsfile = filename.replace(".png",".txt");
	f = open(labelsfile)
	for line in f:
	    labellinelist.append([int(i) for i in line.split()])
	f.close();

	if len(imlist) != len(labellinelist):
		print "Error !!. files not compatible !!. imlist ", len(imlist), "lablelist", labellinelist;
		exit;
	imgListTuple=zip(imlist,labellinelist);

	return imgListTuple;


