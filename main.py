import cv2
import numpy as np
import dlib


def extracting_ind(nparray):
	index=None
	for num in nparray[0]:
		index=num
		break
	return index


img1 = cv2.imread("Cillian_Murphy1.jpeg")

img_gray=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
mask=np.zeros_like(img_gray)

detector=dlib.get_frontal_face_detector()
faces=detector(img_gray)
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
for face in faces:
	landmarks=predictor(img_gray, face)
	landmarks_xy=[]
	for n in range(0,68):
		x=landmarks.part(n).x
		y=landmarks.part(n).y
		landmarks_xy.append((x,y))

		#cv2.circle(img1, (x,y), 1, (0,0,255), -1)
	xy=np.array(landmarks_xy, np.int32)
	hull=cv2.convexHull(xy)
	#cv2.polylines(img1, [hull], True, (255,0,0), 3)
	#cv2.fillConvexPoly(mask, hull, 255)
	facecut1=cv2.bitwise_and(img1, img1, mask=mask)

	#DELAUNAY TRANSFORMATION
	rect=cv2.boundingRect(hull)
	#(x,y,w,h)=rect
	#cv2.rectangle(img1, (x,y), (x+w,y+h), (0,255,0))
	triangledivs=cv2.Subdiv2D(rect)
	triangledivs.insert(landmarks_xy)
	triangles=triangledivs.getTriangleList()
	triangles=np.array(triangles, dtype=int)
	#print(triangles)

	indices_triangles=[]

	indices_triangles_for_RT=[]

	for t in triangles:
		vertone=(t[0],t[1])
		verttwo=(t[2],t[3])
		vertthree=(t[4],t[5])

		#cv2.line(img1, vertone, verttwo, (0,0,255), 1)
		#cv2.line(img1, verttwo, vertthree, (0,0,255), 1)
		#cv2.line(img1, vertthree, vertone, (0,0,255), 1)

		index1=np.where((xy==vertone).all(axis=1))
		index2=np.where((xy==verttwo).all(axis=1))
		index3=np.where((xy==vertthree).all(axis=1))
		#print(index1)
		#print(vertone)
		index1=extracting_ind(index1)
		index2=extracting_ind(index2)
		index3=extracting_ind(index3)
		#print(index1)
		if index1 is not None and index2 is not None and index3 is not None:
			triangle=[index1,index2, index3]
			indices_triangles.append(triangle)
			#indices_triangles_for_RT.append(triangle)


	#print(indices_triangles)

img2 = cv2.imread("Johnny_Depp.jpg")
img2_gray=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
faces2=detector(img2_gray)

ht, wt, channels = img2.shape
img2_deepfake=np.zeros((ht, wt, channels), np.uint8)


for face2 in faces2:
	landmarks2=predictor(img2_gray, face2)
	landmarks2_xy=[]
	for n2 in range(0,68):
		x2=landmarks2.part(n2).x
		y2=landmarks2.part(n2).y
		landmarks2_xy.append((x2,y2))
		#cv2.circle(img2, (x2,y2), 1, (0,0,255), -1)

	xy2=np.array(landmarks2_xy, np.int32)
	hull2=cv2.convexHull(xy2)	

lineSpace_mask=np.zeros_like(img_gray)
lineSpace_df=np.zeros_like(img2)

#Triangulation of second using first

for t in indices_triangles:

	#1
	one1=landmarks_xy[t[0]]
	two1=landmarks_xy[t[1]]
	three1=landmarks_xy[t[2]]

	triangle1_1=np.array([one1, two1, three1], np.int32)
	rect1_1=cv2.boundingRect(triangle1_1)
	(x,y,w,h)=rect1_1
	#cv2.rectangle(img1, (x,y), (x+w, y+h), (0,255,0), 1)
	cropped_triangle=img1[y:y+h, x:x+w]

	cropped_triangle_mask=np.zeros((h,w), np.uint8)


	points=np.array([[one1[0]-x, one1[1]-y], [two1[0]-x, two1[1]-y], [three1[0]-x, three1[1]-y]], np.int32)
	cv2.fillConvexPoly(cropped_triangle_mask, points, 255)
	cropped_triangle=cv2.bitwise_and(cropped_triangle, cropped_triangle, mask=cropped_triangle_mask)


	#cv2.line(img1, one1, two1, (0,0,255), 1)
	#cv2.line(img1, two1, three1, (0,0,255), 1)
	#cv2.line(img1, three1, one1, (0,0,255), 1)	

	lineSpace=cv2.bitwise_and(img1, img1, mask=lineSpace_mask)

	#2
	one2=landmarks2_xy[t[0]]
	two2=landmarks2_xy[t[1]]
	three2=landmarks2_xy[t[2]]

	triangle2_1=np.array([one2, two2, three2], np.int32)
	rect2_1=cv2.boundingRect(triangle2_1)
	(x,y,w,h)=rect2_1
	#cv2.rectangle(img2, (x,y), (x+w, y+h), (0,255,0), 1)
	cropped_triangle2=img2[y:y+h, x:x+w]

	cropped_triangle_mask2=np.zeros((h,w), np.uint8)


	points2=np.array([[one2[0]-x, one2[1]-y], [two2[0]-x, two2[1]-y], [three2[0]-x, three2[1]-y]], np.int32)
	cv2.fillConvexPoly(cropped_triangle_mask2, points2, 255)
	cropped_triangle2=cv2.bitwise_and(cropped_triangle2, cropped_triangle2, mask=cropped_triangle_mask2)


	#cv2.line(img2, one2, two2, (0,0,255), 1)
	#cv2.line(img2, two2, three2, (0,0,255), 1)
	#cv2.line(img2, three2, one2, (0,0,255), 1)

	#Warping

	p1=np.float32(points)
	p2=np.float32(points2)

	Matrix=cv2.getAffineTransform(p1, p2)
	#print(Matrix)

	warpedTri=cv2.warpAffine(cropped_triangle, Matrix, (w, h))
	warpedTri=cv2.bitwise_and(warpedTri, warpedTri, mask=cropped_triangle_mask2)
	

	#Reconstructing

	triangle_Area=img2_deepfake[y:y+h, x:x+w]
	triangle_Area_gray=cv2.cvtColor(triangle_Area, cv2.COLOR_BGR2GRAY)
	_, bg=cv2.threshold(triangle_Area_gray, 1, 255, cv2.THRESH_BINARY_INV)
	
	warpedTri=cv2.bitwise_and(warpedTri, warpedTri, mask=bg)

	triangle_Area=cv2.add(triangle_Area, warpedTri)
	img2_deepfake[y:y+h, x:x+w]=triangle_Area

	#img2_newDF=cv2.cvtColor(img2_deepfake, cv2.COLOR_BGR2GRAY)
	
	#bg=cv2.bitwise_and(img2, img2, mask=bg)
	break
	

img2_face_mask = np.zeros_like(img2_gray)
img2_head_mask=cv2.fillConvexPoly(img2_face_mask, hull2, 255)
img2_face_mask=cv2.bitwise_not(img2_head_mask)

img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)

finalResult=cv2.add(img2_head_noface, img2_deepfake)


(x,y,w,h)=cv2.boundingRect(hull2)
centerFace2=(int((x+x+w)/2), int((y+y+h)/2))
seamlessClone=cv2.seamlessClone(finalResult, img2, img2_head_mask, centerFace2, cv2.NORMAL_CLONE)

'''
realTimeCapture=cv2.VideoCapture(0)


while True:
	_, img3=realTimeCapture.read()
	img3_gray=cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
	img3_deepfake=np.zeros_like(img3)

	

	faces3=detector(img3_gray)
	for face in faces3:
		landmarks3=predictor(img3_gray, face)
		landmarks3_xy=[]
		for n in range(0, 68):
			x=landmarks3.part(n).x	
			y=landmarks3.part(n).y
			landmarks3_xy.append((x,y))

			#cv2.circle(img3, (x,y), 3, (0,255,0), -1)
			#cv2.imshow("Test", img3)

		points3=np.array(landmarks3_xy, np.int32)
		convexHull3=cv2.convexHull(points3)

	lineSpaceMask3=np.zeros_like(img_gray)
	lineSpaceMask3DF=np.zeros_like(img3)

	#TRIANGULATION

	for t in indices_triangles_for_RT:
		rt1=landmarks_xy[t[0]]
		rt2=landmarks_xy[t[1]]
		rt3=landmarks_xy[t[2]]

		triangle_rt_1=np.array([rt1, rt2, rt3], np.int32)

		rect_rt_1=cv2.boundingRect(triangle_rt_1)
		(x,y,w,h)=rect_rt_1
		cropped_triangle_rt=img1[y:y+h, x:x+w]
		cropped_triangle_mask_rt_1=np.zeros((h, w), np.uint8)

		points_rt_1=np.array([[rt1[0]-x, rt1[1]-y], [rt2[0]-x, rt2[1]-y], [rt3[0]-x, rt3[1]-y]], np.int32)

		cv2.fillConvexPoly(cropped_triangle_mask_rt_1, points_rt_1, 255)


		#TRIANGULATION of RT
		rt3_1=landmarks3_xy[t[0]]
		rt3_2=landmarks3_xy[t[1]]
		rt3_3=landmarks3_xy[t[2]]
		triangle3=np.array([rt3_1, rt3_2, rt3_3], np.int32)

		rect_rt_3=cv2.boundingRect(triangle3)
		(x,y,w,h)=rect_rt_3

		cropped_triangle_mask_rt_3=np.zeros((h,w), np.uint8)
		points_rt_3=np.array([[rt3_1[0]-x, rt3_1[1]-y], [rt3_2[0]-x, rt3_2[1]-y], [rt3_3[0]-x, rt3_3[1]-y]], np.int32)

		cv2.fillConvexPoly(cropped_triangle_mask_rt_3, points_rt_3, 255)		


		#WARPING RT
		points_rt_1=np.float32(points_rt_1)
		points_rt_3=np.float32(points_rt_3)

		Matrix_rt=cv2.getAffineTransform(points_rt_1, points_rt_3)
		warpedTri_rt=cv2.warpAffine(cropped_triangle_rt, Matrix_rt, (w,h))
		warpedTri_rt=cv2.bitwise_and(warpedTri_rt, warpedTri_rt, mask=cropped_triangle_mask_rt_3)

		#Reconstruction_RT

		img3_deepfake_rect_area=img3_deepfake[y:y+h, x:x+w]
		img3_deepfake_rect_area_gray=cv2.cvtColor(img3_deepfake_rect_area, cv2.COLOR_BGR2GRAY)

		_, bg_rt=cv2.threshold(img3_deepfake_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
		warpedTri_rt=cv2.bitwise_and(warpedTri_rt, warpedTri_rt, mask=bg_rt)

		img3_deepfake_rect_area=cv2.add(img3_deepfake_rect_area, warpedTri_rt)
		img3_deepfake[y:y+h, x:x+w]=img3_deepfake_rect_area

	#Swapping_RT

	img3_face_mask=np.zeros_like(img3_gray)
	img3_head_mask=cv2.fillConvexPoly(img3_face_mask, convexHull3, 255)
	img3_face_mask=cv2.bitwise_not(img3_head_mask)

	img3_head_noface=cv2.bitwise_and(img3, img3, mask=img3_face_mask)
	RT_result=cv2.add(img3_head_noface, img3_deepfake)

	(x, y, w, h)=cv2.boundingRect(convexHull3)
	centerFace_rt=(int((x+x+w)/2), int((y+y+h)/2))

	seamlessClone_rt=cv2.seamlessClone(RT_result, img3, img3_head_mask, centerFace_rt, cv2.MIXED_CLONE)

	cv2.imshow("img3", img3)
	cv2.imshow("clone", seamlessClone_rt)
	cv2.imshow("res", RT_result)


	key=cv2.waitKey(1)
	if key==27:
		break

'''

cv2.imshow("1", cv2.resize(img1, (400, 400)))
#cv2.imshow("Mask", cv2.resize(facecut1, (400, 400)))
cv2.imshow("2", cv2.resize(img2, (400, 400)))
cv2.imshow("c1", cropped_triangle)
cv2.imshow("c2", cropped_triangle2)
cv2.imshow("maskcropped", cropped_triangle_mask)
#cv2.imshow("maskcropped2", cropped_triangle_mask2)
cv2.imshow("warpedTri", warpedTri)
#cv2.imshow("DF", img2_deepfake)
#cv2.imshow("bg", bg)
#cv2.imshow("RES", finalResult)
#cv2.imshow("seamlessClone", seamlessClone)




cv2.waitKey(0)
cv2.destroyAllWindows()