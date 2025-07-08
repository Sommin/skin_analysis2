import streamlit as st
import os
import sys, base64
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import cv2
import mediapipe as mp

if "face" not in st.session_state:
    st.session_state["face"] = None  # 또는 기본값으로 설정할 다른 값

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


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # PyInstaller 실행 경로
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image(resource_path("resource/IMG_3621.PNG"))
st.set_page_config(
    page_title = 'Multipape App'
)

st.title('Facial Skin Analysis WebCam Demo')
st.sidebar.success('Select a page above after taking photo')

st.markdown(
    f"""
    <div style="text-align: center; margin-bottom: 5px;">
        <img src="data:image/png;base64,{logo_base64}" width="450"><br>
        <h1 style="margin-bottom: 0px; margin-top: 10px;">Emma Healthcare</h1>
        <p style="color: gray; font-size: 16px; margin-top: 1px;">
            Innovation in the AI based Biomedical Technology
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

class FrameSaver(VideoTransformerBase):
    def __init__(self):
        self.latest_frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.latest_frame = img  # 최근 프레임 저장
        return img # pred_mask

ctx = webrtc_streamer(
    key="skin-analysis",
    video_processor_factory=FrameSaver,
    media_stream_constraints={"video": True, "audio": False}
)

if "face_captured" not in st.session_state:
    st.session_state["face_captured"] = False

st.markdown(
    "<p style='text-align: center; font-size: 20px;'>😊 <strong>얼굴을 카메라에 안에 모두 들어가도록 맞춰주세요</p>",
    unsafe_allow_html=True
)
# 가운데 배치: 좌우 여백 + 중앙 버튼
left, center, right = st.columns([2, 1, 2])
with center:
    if st.button("📸 촬영하기"):
        if (
            ctx is None or
            not hasattr(ctx, "video_processor") or
            ctx.video_processor is None or
            ctx.video_processor.latest_frame is None or
            crop_face(ctx.video_processor.latest_frame) is None
        ):
            st.markdown(
                """
                <div style='text-align: left; margin-top: 10px;'>
                    <span style='color: red; font-weight: bold; font-size: 16px; white-space: nowrap;'>
                        ❗ START 버튼을 누르고 다시 촬영해주세요.
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            frame = ctx.video_processor.latest_frame
            st.session_state["face"] = frame
            st.session_state["face_captured"] = True 
if st.session_state["face_captured"]:
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>📷 촬영이 완료되었습니다!</h5>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([3,2,3])
    with col2:
        if st.button("📊 분석 결과 보기"):
            with st.spinner("⚙️ 분석 결과로 이동 중입니다..."):
                time.sleep(1) 
                st.switch_page("pages/2_🔬_skin measure.py")
