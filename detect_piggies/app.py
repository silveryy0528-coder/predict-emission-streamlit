import cv2
import os
from ultralytics import YOLO
from collections import defaultdict
import streamlit as st
import tempfile
import time
import numpy as np


@st.cache_resource
def load_model():
    return YOLO('model.pt')


def infer_frame(frame, conf_threshold):
    model = load_model()
    results = model.track(frame, classes=[0], verbose=False, conf=conf_threshold, iou=0.4)
    return results, model.names


def annotate_frame(frame, results, class_names, appear, id_map, next_id):
    annotated_frame = frame.copy()
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.numpy()
        ids = results[0].boxes.id.numpy()

        cls_id = int(results[0].boxes.cls[0].numpy())
        cls_name = class_names[cls_id]
        conf = results[0].boxes.conf[0].numpy()

        for box, oid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)

            appear[oid] += 1

            # YOLO's internal ID oid may be unstable, assign new stable ID sid after X frames of appearance.
            if appear[oid] >= 10 and oid not in id_map:
                id_map[oid] = next_id
                next_id += 1

            if oid in id_map:
                sid = id_map[oid]
                label = f'{cls_name} ID:{sid} {conf:.2f}'
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 255), 8)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
                cv2.putText(
                    annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 8, cv2.LINE_AA)
                cv2.putText(
                    annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 4, cv2.LINE_AA)

    return annotated_frame


# -----
st.title('Find Neo (YOLO + Streamlit)')

conf_threshold = st.slider('Confidence Threshold:', 0.0, 1.0, 0.5, 0.05)

option = st.radio('Choose input source:', ['Upload Video', 'Webcam'])

if option == 'Upload Video':
    uploaded_video = st.file_uploader('Upload a video', type=['mp4'])

    if uploaded_video is not None:
        # uploaded_video is a file-like object in memory, not a real file on disk
        # However, OpenCV needs a real path and we need a tempfile for this.
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)
        status_text = st.empty()

        stframe = st.empty()
        id_map = {}
        next_id = 1
        appear = defaultdict(int)

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

        display_interval = 5
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            progress = frame_count / total_frames

            results, class_names = infer_frame(frame, conf_threshold)
            annotated_frame = annotate_frame(frame, results, class_names, appear, id_map, next_id)

            out.write(annotated_frame)
            if frame_count % display_interval == 0 or frame_count == total_frames:
                stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels='RGB')
                progress_bar.progress(frame_count / total_frames)
                status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress*100:.1f}%)")


        out.release()
        cap.release()
        with open(out_file, "rb") as f:
            st.download_button(
                label="Download Annotated Video",
                data=f,
                file_name="annotated_video.mp4",
                mime="video/mp4"
            )
        if os.path.exists(tfile.name):
            os.remove(tfile.name)
            print('Delete temporary uploaded file.')
        if os.path.exists(out_file):
            os.remove(out_file)
            print('Delete temporary downloaded file.')


elif option == 'Webcam':
    st.write('Click start to capture from webcam')
    camera_input = st.camera_input('Take a picture or record a short video')

    if camera_input is not None:
        file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        id_map = {}
        next_id = 1
        appear = defaultdict(int)

        results, class_names = infer_frame(frame, conf_threshold)
        annotated_frame = annotate_frame(frame, results, class_names, appear, id_map, next_id)

        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels='RGB')

