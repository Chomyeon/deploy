import cv2
from flask import Flask, render_template, Response, request, jsonify
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import *
from utils.preprocessor import preprocess_input
from gaze_tracking import GazeTracking
import json

# 아이트래커 준비
gaze = GazeTracking()

# 표정 감성 인식
USE_WEBCAM = True # If false, loads video file source

# parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting video streaming

cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

app = Flask(__name__, static_url_path='/static')
cap = cv2.VideoCapture(0)
total_frame = 0
nerv_frame = 0

def generate_frames():
    global total_frame
    global nerv_frame
    nerv_gage = 0
    ud_gage = 0
    nerv_flag = 0
    ud_flag = 0
    while cap.isOpened(): # True:
        
        total_frame += 1
        ret, bgr_image = cap.read()
        
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
                minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
        for face_coordinates in faces:

            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
        
            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
                emotion_window.append("표정이 좋지 않습니다. 긴장을 풀어주세요.")
                nerv_gage += 6
                nerv_frame += 1
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
                emotion_window.append("표정이 좋지 않습니다. 긴장을 풀어주세요.")
                nerv_gage += 6
                nerv_frame += 1
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
                emotion_window.append("좋습니다!")
                nerv_gage -= 1
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
                emotion_window.append("")
            else:
                color = emotion_probability * np.asarray((0, 255, 0))
                nerv_gage += 1
                nerv_frame += 1
                emotion_window.append("")
            
            tcolor = []
            for i in range(3):
                tcolor.append(color[i])
            color = tuple(tcolor)
            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                continue

            draw_bounding_box(face_coordinates, rgb_image, color)
            rgb_image = draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -45, font_scale=2)
            
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR) 
        
        # 아이트래킹 코드
        gaze.refresh(bgr_image)
        text = ""
        bgr_image = gaze.annotated_frame()
        if nerv_gage >= 100:
            nerv_gage = 0
            nerv_flag = 10
        if nerv_flag >= 0:
            text = "표정이 좋지 않습니다. 긴장을 풀어주세요"
            nerv_flag -= 1
            nerv_gage = 0
        
        
        if (gaze.is_up_down()):
            ud_gage += 1
        ud_text = ""
       
        if total_frame == 30:
            ud_gage = 0
            ud_flag = 10
        if ud_flag >= 0:
            ud_text = "정면을 바라봐 주세요"
            ud_flag -= 1
            ud_gage = 0
        font_path = '/Users/im_jungwoo/Desktop/project/khuthon/face-recog/utils/NanumGothic.ttf'
        bgr_image = draw_text_imgdraw(bgr_image, ud_text, (90,60), font_path, font_size = 50, text_color=(144, 238, 144),stroke_width = 2)

        # 텍스트 크기 얻기
        font_size = 30
        font = ImageFont.truetype(font_path, font_size)
      
        pil = Image.fromarray(bgr_image)
        draw = ImageDraw.Draw(pil)
        
        
        # 함수 호출
        text_position = ((bgr_image.shape[1] - draw.textlength(text, font)) // 2, bgr_image.shape[0] - 50)
        draw_text_with_rectangle(draw, text, position=(text_position[0],text_position[1]), font_path = font_path)

        bgr_image = np.array(pil)
        
        _, buffer = cv2.imencode('.jpg', bgr_image)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/send_url', methods=['GET'])
def send_url():
    # GET 요청에서 전달된 쿼리 파라미터를 받아오기
    url_parameter = request.args.get('url_parameter')
    percent = str(int((nerv_frame / max(1, total_frame) * 100)))
    response_data = {'tension': percent}
    return jsonify(response_data)

@app.route('/static/result.html')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True, port = 8000)

