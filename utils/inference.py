import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from PIL import Image, ImageDraw, ImageFont

def load_image(image_path, grayscale=False, target_size=None):
    pil_image = image.load_img(image_path, grayscale, target_size)
    return image.img_to_array(pil_image)

def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    color = tuple(map(int, color))  # Convert color to tuple of integers
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)


def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
              font_scale=2, thickness=2):
    font_size = int(10 * font_scale)
    font = ImageFont.truetype('/Users/im_jungwoo/Desktop/project/khuthon/face-recog/utils/NanumGothic.ttf', font_size)
    pil = Image.fromarray(image_array)
    draw = ImageDraw.Draw(pil)
    x, y = coordinates[:2]

    color = tuple(map(int, color))  # Convert color to tuple of integers
    draw.text((x + x_offset, y + y_offset), text, color, font=font)
    image_array = np.array(pil)
    return image_array


def get_colors(num_classes):
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    colors = np.asarray(colors) * 255
    return colors

def draw_text_with_rectangle(draw, text, position, font_path, text_color=(255, 255, 255), rectangle_color=(0, 0, 0)):
    # 폰트 설정
    font = ImageFont.truetype(font_path, 30)

    # 텍스트의 크기 가져오기
    text_size = draw.textlength(text, font)
    
    text_position = (position[0], position[1])

     # 직사각형의 좌표 계산 
    rectangle_position = (position[0] - 10, position[1], position[0] + int(text_size) + 10, position[1] + 100)

    # 직사각형 그리기
    draw.rectangle(rectangle_position, fill=rectangle_color)

    # 텍스트 그리기
    draw.text(text_position, text, font=font, fill=text_color)

def draw_text_imgdraw(image, text, position, font_path, font_size, text_color, stroke_width):
        # Image 객체로 변환
    pil_image = Image.fromarray(image)

    # ImageDraw 객체 생성
    draw = ImageDraw.Draw(pil_image)

    # 폰트 설정
    font = ImageFont.truetype(font_path, font_size)

    # 텍스트 그리기
    draw.text(position, text, font=font, fill=text_color, stroke_width=stroke_width)

    # PIL Image를 다시 numpy.ndarray로 변환
    result_image = np.array(pil_image)

    return result_image
