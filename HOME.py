import streamlit as st
from streamlit_option_menu import option_menu
from train import show_training_form
from deteksi import deteksi_page

st.set_page_config(
    page_title="Deteksi Plat Kendaraan",
    page_icon="🚘",
)

def main():

    with st.sidebar:
        # st.title("Menu Utama")
        # Sidebar menu
        selected_option = option_menu("VIPER-Sys Menu:", ["Beranda", "Trainning Model", "Recognition & OCR"])

    # Tampilkan halaman yang dipilih
    if selected_option == "Beranda":
        st.title("Selamat datang di VIPER-Sys")
        st.write("Sistem ini dilengkapi dengan tranning model menggunakan YOLOv5, dan bisa menghaislkan tulisan hasil pengenalan karakter menggunakan Easy OCR")
        st.write("**Menu Trainning**")
        st.markdown(
            "- Terdapat inputan zip file datset pelatihan yang telah dilabeli pada Roboflow\n"
            "- Melakukan Training dengan menentukan nilai batch size dan epoch\n"
            "- Menyimpan model yang telah ditraining"
        )
        st.write("***Disclaimer*** : Disarankan pelatihan model dilakukan pada *Google Colab* dikarenakan terdapat free GPU, dengan menggunakan API dataset dari roboflow berikut:")
        expander = '''curl -L "https://app.roboflow.com/ds/LDgWMBMWFJ?key=zLce4owNZI" &gt; 
roboflow.zip; unzip roboflow.zip; rm roboflow.zip'''
        st.code(expander, 'python')
        st.write("Berikut link pelatihan menggunakan *Google Colab*")
        expanders = '''https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb'''
        st.code(expanders, 'python')
        st.write("**Menu Recognition & OCR**")
        st.markdown(
            "- Melakukan input model yang telah dilatih\n"
            "- Menginputkan gambar yang ingin dideteksi\n"
            "- Menampilkan hasil deteksi dan pengenalan karakter"
        )
    elif selected_option == "Trainning Model":
        show_training_form()
    elif selected_option == "Recognition & OCR":
        deteksi_page()

if __name__ == "__main__":
    main()
