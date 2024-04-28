from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from PIL import Image, ImageDraw
import face_recognition
import os

app = Flask(__name__)

# Directory paths
UPLOAD_FOLDER = 'uploads'
RECOGNIZED_FOLDER = 'recognized'

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # If no file is selected
        if file.filename == '':
            return redirect(request.url)

        # If file is selected
        if file:
            # Save the uploaded file
            filename = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filename)

            # Perform face recognition
            image = face_recognition.load_image_file(filename)
            face_locations = face_recognition.face_locations(image)
            if len(face_locations) == 0:
                return render_template('error.html', message='No faces detected.')
            else:
                # Assuming only one face is detected
                top, right, bottom, left = face_locations[0]
                
                # Crop the detected face
                face_image = image[top:bottom, left:right]

                # Save the cropped face image
                cropped_face_path = os.path.join(UPLOAD_FOLDER, 'cropped_face.jpg')
                cropped_face = Image.fromarray(face_image)
                cropped_face.save(cropped_face_path)

                # Perform face recognition for known faces
                face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                known_encodings = []
                known_names = []

                # Load known encodings and names
                for file_name in os.listdir(RECOGNIZED_FOLDER):
                    img = face_recognition.load_image_file(os.path.join(RECOGNIZED_FOLDER, file_name))
                    known_encoding = face_recognition.face_encodings(img)[0]
                    known_encodings.append(known_encoding)
                    known_names.append(os.path.splitext(file_name)[0])

                # Compare the face encoding with known encodings
                matches = face_recognition.compare_faces(known_encodings, face_encoding)

                # Draw a green rectangle around the face
                pil_image = Image.fromarray(image)
                draw = ImageDraw.Draw(pil_image)
                draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=3)
                annotated_image_path = os.path.join(UPLOAD_FOLDER, 'annotated_image.jpg')
                pil_image.save(annotated_image_path)

                if True in matches:
                    first_match_index = matches.index(True)
                    recognized_name = known_names[first_match_index]
                    return render_template('result.html', recognized_name=recognized_name)
                else:
                    return render_template('unknown.html', filename=annotated_image_path)

    return render_template('upload.html')


@app.route('/add_name', methods=['POST'])
def add_name():
    if request.method == 'POST':
        filename = request.form['filename']
        name = request.form['name']
        
        # Move the image to the recognized folder with the provided name
        new_filename = os.path.join(RECOGNIZED_FOLDER, name + '.jpg')
        os.rename(filename, new_filename)
        
        return redirect(url_for('upload_image'))

# Define the route to serve images
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    # Return the image file from the 'uploads' folder
    return send_from_directory('', filename)

if __name__ == '__main__':
    app.run(debug=True)
