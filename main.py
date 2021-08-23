import cv2 as cv
import streamlit as st
import numpy as np
import mediapipe as mp
from PIL import Image,ImageEnhance



class Face_detection:
	def __init__(self):
		pass

	def detect_faces(self,up_image):
		'Detect faces using Opencv'

		# cascade files
		facedetect = cv.CascadeClassifier(r'cascade_files\haarcascade_frontalface_default.xml')
		detect_img = np.array(up_image.convert('RGB'))
		new_img  = cv.cvtColor(detect_img,1)
		faces = facedetect.detectMultiScale(new_img,1.3,5)
		for x,y,w,h in faces:
			cv.rectangle(new_img,(x,y),(x+w,y+h),(255,255,0),2)
		return new_img,faces

	def detect_eyes(self,up_image):
		'Detect eyes using Opencv'
		eyes = cv.CascadeClassifier(r'cascade_files\haarcascade_eye.xml')
		detect_img = np.array(up_image.convert('RGB'))
		new_img1 = cv.cvtColor(detect_img,1)
		both_eye = eyes.detectMultiScale(new_img1,1.3,5)
		for x,y,w,h in both_eye:
			cv.rectangle(new_img1,(x,y),(x+w,y+h),(255,255,0),2)
		return new_img1,both_eye

	def facial_landmarks(self,up_image):
		" Facial landmarks using google's mediapipe "
		# face mesh
		face_mesh = mp.solutions.face_mesh
		face_mesh = face_mesh.FaceMesh()

		img = np.array(up_image.convert('RGB'))

		#Facial landmarks
		result = face_mesh.process(img)

		height, width, _ = img.shape

		for facial_landmarks in result.multi_face_landmarks:
			for i in range(0,468):
				pt1 = facial_landmarks.landmark[i]
				x = int(pt1.x * width)
				y = int(pt1.y * height)

				cv.circle(img , (x,y), 2, (50,100,250), -1)

		return img

	#Function for defining streamlit webapp
	def main(self):
		st.title("Face Detection App")
		st.write("Build with streamlit,opencv and python")
		activities= ["Detection","About"]
		choices = st.sidebar.selectbox("Select Activities",activities)

		if choices=="Detection":
			st.subheader("Face Detection")
			img_file =st.file_uploader("Upload_file",type=["png","jpg","jpeg"])  
			if img_file is not None:
				up_image = Image.open(img_file)
				st.image(up_image)
			enhance_type = st.sidebar.radio("Enhance type",["Original","Gray-scale","Contrast","Brightness","Blurring"])
			if enhance_type == "Gray-scale":
				new_img = np.array(up_image.convert('RGB'))
				img = cv.cvtColor(new_img,1)
				gray = cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)
				st.image(gray)
			if enhance_type=="Contrast":
				c_make = st.sidebar.slider("Contrast",0.5,3.5)
				enhancer = ImageEnhance.Contrast(up_image)
				img_out = enhancer.enhance(c_make)
				st.image(img_out)
			if enhance_type=="Brightness":
				b_make = st.sidebar.slider("Brightness",0.5,3.5)
				enhancer = ImageEnhance.Brightness(up_image)
				img_bg = enhancer.enhance(b_make)
				st.image(img_bg)
			if enhance_type=="Blurring":
				br_make = st.sidebar.slider("Blurring",0.5,3.5)
				br_img = np.array(up_image.convert('RGB'))
				b_img = cv.cvtColor(br_img,1)
				blur = cv.GaussianBlur(b_img,(11,11),br_make)
				st.image(blur)
			task=["Faces","Eye","Facial_Landmarks"]
			feature_choice = st.sidebar.selectbox("Find Feature",task)

			if st.button("Process"):
				if feature_choice == "Faces":
					result_img,result_faces = self.detect_faces(up_image)
					st.image(result_img)
					st.success("Found {} faces.".format(len(result_faces)))
				if feature_choice == "Eye":
					result_img,result_eyes = self.detect_eyes(up_image)
					st.image(result_img)
					st.success("Found {} eyes.".format(len(result_eyes)))
				if feature_choice == "Facial_Landmarks":
					result_img = self.facial_landmarks(up_image)
					st.image(result_img)
					st.success("Found {} face for landmark in image.".format(len(result_img)//480))



		elif choices =="About":
			st.write("This Application is Developed By Aman Jain")
			st.write("Thank you for visiting")



if __name__=='__main__':
	detection = Face_detection()
	detection.main()



