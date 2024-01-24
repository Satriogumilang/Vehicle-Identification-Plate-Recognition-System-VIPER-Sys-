import os
import streamlit as st
import shutil
import subprocess
import zipfile
import torch
from IPython.display import Image, clear_output  # to display images

print(
    f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

# Fungsi untuk mengekstraksi file ZIP


def extract_zip(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# Fungsi untuk menjalankan training YOLOv5


def train_yolov5(data_path, batch_size, epochs):
    # Ganti direktori ke folder YOLOv5
    os.chdir("yolov5")

    # Jalankan training dengan command line
    command = f"python train.py --img 416 --batch {batch_size} --epochs {epochs} --data data.yaml --weights yolov5x.pt --cache"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, universal_newlines=True)

    # Tampilkan keterangan saat training berlangsung
    st.text("Training in progress...")
    for line in process.stdout:
        st.text(line.strip())

    # Tampilkan keterangan saat training selesai
    st.text("Training completed!")

# Fungsi untuk menyimpan model hasil training


def save_model():
    # Path to the weights file
    weights_file = './runs/train/exp5/weights/best.pt'

    # Destination path for saving the weights file
    destination_path = '../new_model.pt'

    shutil.copy2(weights_file, destination_path)


# Fungsi untuk menampilkan form dan tombol
def show_training_form():
    # Tampilan Streamlit
    st.title("YOLOv5 Training")

    # Input dataset dalam format .zip
    data_zip = st.file_uploader("Upload dataset ZIP file", type="zip")

    # Input batch size dan epoch
    batch_size = st.number_input("Batch Size", min_value=1, value=16)
    epochs = st.number_input("Epochs", min_value=1, value=50)

    # Tombol untuk memulai training
    if st.button("Start Training") and data_zip is not None:
        # Simpan file ZIP dataset ke disk
        with open("data.zip", "wb") as f:
            f.write(data_zip.getbuffer())

        # Ekstraksi dataset dan simpan di folder "datasets"
        extract_zip("data.zip", "datasets")

        # Jalankan training YOLOv5
        train_yolov5("datasets", batch_size, epochs)

    # Tombol untuk menyimpan model hasil training
    if st.button("Save Model"):
        save_model()
        st.success("Model saved successfully!")


