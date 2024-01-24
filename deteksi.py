import cv2
import torch
from time import time
import numpy as np
import streamlit as st
import easyocr

class DeteksiObjek:
    def __init__(self):
        self.model_path = None
        self.model = None
        self.classes = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using Device: ", self.device)

    def set_model_path(self, model_path):
        self.model_path = model_path
        self.model = self.load_model(self.model_path)
        self.classes = self.model.names

    def load_model(self, model_path):
        """
        load yolov5 model
        """
        model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxy[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        label_colors = {
            "plat": (0, 0, 255),  # Merah
        }

        for i in range(n):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = (
                    int(row[0] * x_shape),
                    int(row[1] * y_shape),
                    int(row[2] * x_shape),
                    int(row[3] * y_shape),
                )
                label = self.class_to_label(labels[i])
                bgr = label_colors.get(label, (0, 0, 0))

                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    bgr,
                    2,
                )

        return frame

    def process_image(self, image_file):
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame = cv2.resize(frame, (640, 640))

        start_time = time()
        results = self.score_frame(frame)
        frame = self.plot_boxes(results, frame)

        # Crop the detected plate region
        plat_results = [result for result in results[0] if int(result) == 0]
        if plat_results:
            selected_plate = plat_results[0]
            plate_cord = results[1][int(selected_plate.item()), :4].cpu().numpy().astype(int)
            cropped_plate = frame[plate_cord[1]:plate_cord[3], plate_cord[0]:plate_cord[2]]

            # Perform OCR on the cropped plate
            ocr_result = self.perform_ocr(cropped_plate)

            # # Display the OCR result
            # st.subheader("OCR Result:")
            # st.write(ocr_result)

            # Draw bounding box for the cropped plate
            cv2.rectangle(frame, (plate_cord[0], plate_cord[1]), (plate_cord[2], plate_cord[3]), (255, 0, 0), 2)

        end_time = time()
        fps = 1 / np.round(end_time - start_time, 2)
        print(f"Frames Per Second : {fps}")

        cv2.putText(
            frame,
            "",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            2,
        )

        return frame

    def perform_ocr(self, cropped_plate):
        reader = easyocr.Reader(['en'])
        results = reader.readtext(cropped_plate)

        ocr_text = ""
        for detection in results:
            ocr_text += f"Text: {detection[1]}, Confidence: {detection[2]:.2f}\n"

        return ocr_text

def deteksi_page():
    st.title("YOLOv5 Object Detection and OCR with Streamlit")

    uploaded_model = st.file_uploader("Choose a YOLOv5 model file...", type=["pt"])

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_model is not None and uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Get the path of the uploaded model or use a default value
        model_path = uploaded_model.name if uploaded_model else "best.pt"

        deteksi_objek = DeteksiObjek()
        deteksi_objek.set_model_path(model_path)
        processed_frame = deteksi_objek.process_image(uploaded_image)

        # st.image(processed_frame, channels="BGR", caption="Processed Image", use_column_width=True)

        # Perform OCR and display the result as text
        ocr_result = deteksi_objek.perform_ocr(processed_frame)

        # Crop the detected plate region
        plat_results = deteksi_objek.score_frame(processed_frame)[0]
        if int(0) in plat_results:
            selected_plate = np.where(plat_results == int(0))[0][0]
            plate_cord = deteksi_objek.score_frame(processed_frame)[1][selected_plate, :4].cpu().numpy().astype(int)
            cropped_plate = processed_frame[plate_cord[1]:plate_cord[3], plate_cord[0]:plate_cord[2]]

            st.subheader("Cropped Plate:")
            st.image(cropped_plate, channels="BGR", caption="Cropped Plate", use_column_width=True)

            st.subheader("OCR Result:")
            st.write(ocr_result)

            

            


