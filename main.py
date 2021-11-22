#loading libraries
from PIL import Image, ImageDraw
import face_recognition
import os
import os.path 

#first part
known_database = './img/known'

known_faces = []
names = []

#a loop through images address
for image_file in os.listdir(known_database):
    #reading the image
    image_path = os.path.join(known_database, image_file)
    image = face_recognition.load_image_file(image_path)

    #feature extraction from the image
    face_encoding = face_recognition.face_encodings(image)[0]
    face_name = image_file.split('.')[0]

    #build our dataset
    known_faces.append(face_encoding)
    names.append(face_name)

#part two

unknown_samples = './img/unknown'

#a loop through images addresses
for image_file in os.listdir(unknown_samples):
    
    #reading the image
    image_path = os.path.join(unknown_samples, image_file)
    image = face_recognition.load_image_file(image_path)
    #make the image ready for drawing
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    #fingding the faces
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        #find the face matches
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        recognized_name = "Unknown Person"
        #finding the label
        if True in matches:
            first_match_index = matches.index(True)
            recognized_name = names[first_match_index]
        #draw rectangle aroung the face with its label
        draw.rectangle(((left, top), (right, bottom)), outline=(255,255,0))
        text_width, text_height = draw.textsize(recognized_name)
        draw.rectangle(((left,bottom - text_height - 10), (right, bottom)), fill=(255,255,0), outline=(255,255,0))
        draw.text((left + 6, bottom - text_height - 5), recognized_name, fill=(0,0,0))
        
    #save the images
    pil_image.save(f'img/recognized/{image_file}.jpg')

    del draw
#show the recognized faces
# pil_image.show()

