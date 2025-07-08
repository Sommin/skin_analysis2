import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from skimage import feature
import time
import math
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as tf_image
import streamlit as st
import base64
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import torch
from resource.unet_model import UNet
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
import os
import sys
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
def load_checkpoint(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

stage1_model = UNet(3,1)
stage2_model = UNet(4,2)
stage1_model = load_checkpoint(stage1_model, resource_path('resource/stage1_unet.pth'), device)
stage2_model = load_checkpoint(stage2_model, resource_path('resource/stage2_unet.pth'), device)
stage1_model.eval()
stage2_model.eval()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils
MODEL_PATH = resource_path('resource/skin_age.h5')
new_model = load_model(MODEL_PATH)

def load_checkpoint(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return model


def compute_lbp_texture(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 2
    n_points = 24
    lbp = feature.local_binary_pattern(gray, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype('float')
    return np.sum(lbp_hist)

def draw_landmarks_with_flicker(image):
    results = face_mesh.process(image)
    landmarks_image = np.zeros_like(image, dtype=np.uint8)
    
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            for connection in mp_face_mesh.FACEMESH_TESSELATION:
                start_idx = connection[0]
                end_idx = connection[1]

                start_point = (int(landmarks.landmark[start_idx].x * image.shape[1]),
                               int(landmarks.landmark[start_idx].y * image.shape[0]))
                end_point = (int(landmarks.landmark[end_idx].x * image.shape[1]),
                             int(landmarks.landmark[end_idx].y * image.shape[0]))

                cv2.line(landmarks_image, start_point, end_point, (220, 220, 220), 1, lineType=cv2.LINE_AA)
                
                # Draw the landmark points
                cv2.circle(landmarks_image, start_point, 1, (127, 127, 127), -1)

    # Now, apply a slight blur to make the lines appear thinner
    landmarks_image = cv2.GaussianBlur(landmarks_image, (3, 3), 0)
    
    # Blend the original image with the landmarks image for a translucent effect
    alpha = 0.35
    blended_image = cv2.addWeighted(image, 1 - alpha, landmarks_image, alpha, 0)
    
    return blended_image

def count_wrinkles_and_spots(roi):
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray_roi, 9, 80, 80)
    edges = cv2.Canny(bilateral, 50, 150)
    
    wrinkles = np.sum(edges > 0)
    
    # Use adaptive thresholding
    thresh1 = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Use morphological operations to fill small holes and remove small noises
    kernel = np.ones((3,3), np.uint8)
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out very small contours to reduce noise
    min_spot_area = 4
    spots = len([cnt for cnt in contours if cv2.contourArea(cnt) > min_spot_area])
    
    return wrinkles, spots


def count_features(image):
    wrinkles, spots = count_wrinkles_and_spots(image)
    texture = compute_lbp_texture(image)
    return wrinkles, spots, texture
def loadImage(image_np):
    # 반드시 180x180x3으로 리사이즈
    resized = cv2.resize(image_np, (180, 180))
    # 정규화 및 차원 확장
    resized = resized.astype('float32') / 255.0
    resized = np.expand_dims(resized, axis=0)  # (1, 180, 180, 3)
    return resized

def model_predict(crop_face):
    global new_model
    age_pred = new_model.predict(loadImage(crop_face))
    x = age_pred[0][0]
    rounded_age_value = round(x)  # Rounds 24.56 to 25
    age = 'About '+ str(rounded_age_value) +' years old'
    return age

def process_image(frame):
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]

    cropped_face, location = crop_face(frame)

    wrinkles, spots, texture = count_features(cropped_face)  # 배경 제거된 얼굴 기준 분석
    frame_with_landmarks = draw_landmarks_with_flicker(cropped_face)  # 전체 이미지 위에 landmark만 덧씌움

    return cropped_face, wrinkles, spots, texture, location


def draw_corner_box(image, location,color=(105, 105, 105)	, thickness=6, length=25):
    img = np.array(image)
    x,y,w,h = location
    # 좌상단
    cv2.line(img, (x, y), (x + length, y), color, thickness)
    cv2.line(img, (x, y), (x, y + length), color, thickness)

    # 우상단
    cv2.line(img, (x + w, y), (x + w - length, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + length), color, thickness)

    # 좌하단
    cv2.line(img, (x, y + h), (x + length, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - length), color, thickness)

    # 우하단
    cv2.line(img, (x + w, y + h), (x + w - length, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - length), color, thickness)

    return img


def crop_face(image):
    with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        results = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            h, w, _ = image.shape
            x_min = int(bbox.xmin * w)
            y_min = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            cropped = image[y_min:y_min+height, x_min:x_min+width]
            return cropped, (x_min, y_min, width, height)
    return image , None # 얼굴이 없으면 원본 반환


def pore_size(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    gray = cv2.resize(gray, (width//4, height//4))

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (3,3), 0)

    # Thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Edge Detection
    edges = cv2.Canny(thresh, 1, 2)

    # Find Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(len(contours))
    # Filter contours
    filtered_contours = []
    areas = []
    for cnt in contours:
        cnt_area = cv2.contourArea(cnt) * 13500/height * 13500/height
        if cnt_area > 100:
            filtered_contours.append(cnt)
            areas.append(cnt_area)
    average_area = np.average(areas)
    biggest_pore = max(areas)
    # Fit Circles
    circles = []
    for cnt in filtered_contours:
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        circles.append((int(x), int(y), int(radius)))

    # Pick the biggest circle
    if len(circles) == 0:
        return image, blurred, edges, average_area
        
    
    biggest_circle = max(circles, key=lambda x: x[2])
    x, y, r = biggest_circle

    # Get the center and the radius of the region of interest
    height, width = image.shape[:2]
    center = (width//2, height//2)
    radius = min(center[0], center[1])

    # Highlight the biggest pore in the region of interest
    if (x*4 + center[0] - radius) - r*4 >= 0 and (y*4 + center[1] - radius) - r*4 >= 0 and (x*4 + center[0] - radius) + r*4 <= 2*radius and (y*4 + center[1] - radius) + r*4 <= 2*radius:
        cv2.circle(image, (x*4 + center[0] - radius, y*4 + center[1] - radius), r*4, (0,255,0), 2)

    return image, blurred, edges, average_area, biggest_pore, areas

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image(resource_path("resource/IMG_3621.PNG"))


def wrinkle(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # → RGB로 변환
    img = cv2.resize(img, (1024, 1024))            # → 사이즈 맞춤

    # 2. NumPy → torch.Tensor (float32로 변환)
    img = img.astype('float32') 
    if img.max() > 1.0:
        img = img/ 255.0 
        img = img*2.0 -1.0    
    img = torch.from_numpy(img).permute(2, 0, 1)   # [H, W, C] → [C, H, W]
    img = img.unsqueeze(0) 

    with torch.inference_mode():
        mask = stage1_model(img.to(device))
        img = torch.concat([img, mask], dim=1)
        pred = stage2_model(img)
        pred_mask = pred.argmax(dim=1)
        pred_mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8)

    return pred_mask

def overaly_wrinkle(image, pred_mask, position):
    x, y, w, h = position
    pred_resized = cv2.resize(pred_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    overlay_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    overlay_mask[y:y+h, x:x+w] = pred_resized  # crop 위치에 붙이기

    dark_region = overlay_mask == 1
    overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    overlay[dark_region] = (overlay[dark_region] * 0.4).astype(np.uint8)
    return overlay


st.title("📋 Facial Skin Analysis Report")
st.markdown("---")
st.session_state['face_captured'] = False

frame = st.session_state.get('face', None)
if frame is None:
    st.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 20px;">
            <img src="data:image/png;base64,{logo_base64}" width="450" style="margin-bottom: 30px;"><br>
            <p style="font-size:18px; margin-top: 20px;">
                <strong>📸 <code>homepage</code>에서 얼굴을 촬영해 주세요!</strong> 😊
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    
if frame is not None:
    try:
        cropped_face, wrinkles, spots, texture, position = process_image(frame)
        age = model_predict(cropped_face)
        _, _, _, average_area, biggest_pore, _ = pore_size(cropped_face)
        average_dia = 2 * math.sqrt(average_area / math.pi)
        biggest_dia = 2 * math.sqrt(biggest_pore / math.pi)
        pred_mask = wrinkle(cropped_face)
        overlay = overaly_wrinkle(frame, pred_mask, position)
        # 결과 저장
        st.session_state.latest_result = {
            "frame": frame,
            "wrinkles": wrinkles,
            "spots": spots,
            "texture": texture,
            "age": age,
            "average_area": average_area,
            "biggest_pore": biggest_pore,
            "average_dia": average_dia,
            "biggest_dia": biggest_dia,
            "overlay": overlay
        }
    except ValueError as e:
        st.error("❗ 오류가 발생했습니다: 입력 마스크 크기가 잘못되었습니다.")
        st.markdown("5초 후 홈으로 자동 이동합니다...")
        time.sleep(5)
        # 메타 태그로 리디렉션
        st.markdown("""
            <meta http-equiv="homepage" content="5; url=/" />
        """, unsafe_allow_html=True)
        st.switch_page("1_🧑‍💻_homepage.py")
        # 실행 중단
        st.stop()        



# 결과 표시
    if st.session_state.latest_result:
        result = st.session_state.latest_result

        st.markdown("### 📸 Wrinkle Face", unsafe_allow_html=True)

        st.image(overlay, caption="Wrinkle Visualization", width=500)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # 👇 피부 지표 요약
        st.markdown("### 🧪 피부 지표 요약")
        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.markdown(f"📅 <b>Predicted Age</b><br><span style='font-size: 22px;'>{age} years</span>", unsafe_allow_html=True)
            st.markdown(f"1️⃣ <b>Wrinkle Count</b><br><span style='font-size: 22px;'>{wrinkles} pixels</span>", unsafe_allow_html=True)
            st.markdown(f"2️⃣ <b>Spots Count</b><br><span style='font-size: 22px;'>{spots} contours</span>", unsafe_allow_html=True)
            st.markdown(f"3️⃣ <b>Texture Intensity</b><br><span style='font-size: 22px;'>{texture}</span>", unsafe_allow_html=True)

        with metric_col2:
            st.markdown(f"⚪ <b>Avg Pore Diameter</b><br><span style='font-size: 22px;'>{average_dia:.2f} µm</span>", unsafe_allow_html=True)
            st.markdown(f"🔴 <b>Max Pore Diameter</b><br><span style='font-size: 22px;'>{biggest_dia:.2f} µm</span>", unsafe_allow_html=True)
            st.markdown(f"⚪ <b>Avg Pore Area</b><br><span style='font-size: 22px;'>{average_area:.2f} µm<sup>2</sup></span>", unsafe_allow_html=True)
            st.markdown(f"🔴 <b>Max Pore Area</b><br><span style='font-size: 22px;'>{biggest_pore:.2f}  µm<sup>2</sup></span>", unsafe_allow_html=True)
