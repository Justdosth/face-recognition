from PIL import Image
import face_recognition 

image = face_recognition.load_image_file('./img/known/me.jpg')
face_locations = face_recognition.face_locations(image)
# print(face_locations)

for face_location in face_locations:
    top, right, bottom, left = face_location

    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()