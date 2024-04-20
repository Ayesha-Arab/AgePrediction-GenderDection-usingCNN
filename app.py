from flask import Flask, render_template, request, send_from_directory, jsonify, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
loaded_model = load_model(r'C:\Users\gaming\PycharmProjects\Gender-Age _Dectection Model\age_gender_model_final', compile=False)

# Create a temporary directory to store uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Flag to check if the image has been processed
image_processed = False

def get_image_features(file_storage):
    # Save the uploaded file temporarily
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file_storage.filename)
    file_storage.save(temp_path)

    # Process the image using your CNN model
    img = load_img(temp_path, grayscale=True)
    img = img.resize((loaded_model.input_shape[1], loaded_model.input_shape[2]), Image.LANCZOS)
    img = np.array(img)
    img = img.reshape(1, *loaded_model.input_shape[1:])
    img = img / 255.0
    # Remove the temporary file
    os.remove(temp_path)
    return img


@app.route('/')
def index():
    global image_processed
    return render_template('index.html', image_processed=image_processed)

@app.route('/upload', methods=['POST'])
def upload():
    global image_processed
    if request.method == 'POST':
        # Get the uploaded image from the request
        img_to_test = request.files['fileToUpload']
        # Process the image using your CNN model
        features = get_image_features(img_to_test)
        pred = loaded_model.predict(features)
        gender_mapping = {
            1: 'Female',
            0: 'Male'
        }
        gender = gender_mapping[round(pred[0][0][0])]
        age = round(pred[1][0][0])
        # Set the flag to True after processing the image
        image_processed = True
        return render_template('index.html', image_processed=image_processed,result=f'Predicted Age: {age} Predicted Gender: {gender}')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [
                                 104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3]*frameWidth)
            y1 = int(detections[0, 0, i, 4]*frameHeight)
            x2 = int(detections[0, 0, i, 5]*frameWidth)
            y2 = int(detections[0, 0, i, 6]*frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes


def gen_frames():
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    video = cv2.VideoCapture(0)
    padding = 20
    while cv2.waitKey(1) < 0:
        hasFrame, frame = video.read()
        if not hasFrame:
            cv2.waitKey()
            break

        resultImg, faceBoxes = highlightFace(faceNet, frame)
        if not faceBoxes:
            print("No face detected")

        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1]-padding):
                         min(faceBox[3]+padding, frame.shape[0]-1),
                         max(0, faceBox[0]-padding):
                         min(faceBox[2]+padding, frame.shape[1]-1)]

            blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            print(f'Age: {age[1:-1]} years')

            cv2.putText(resultImg, f'{gender}, {age}', (
                faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        ret, encodedImg = cv2.imencode('.jpg', resultImg)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')




@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)