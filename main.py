import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image

# Interface utilisateur Streamlit
st.title("Recherche de Visages Similaires")

uploaded_file = st.file_uploader("Charger une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    files = {"uploaded_file": uploaded_file.getvalue()}
    response = requests.post("http://127.0.0.1:8000/faceID/", files=files)

    if response.status_code == 200:
        data = response.json()
        if "similar_faces" in data:
            st.header("Visages similaires trouvés :")
            for image_path in data["similar_faces"]:
                loaded_image = cv2.imread(image_path)
                if loaded_image is not None:
                    st.image(loaded_image, caption='Image similaire', width=200)
                else:
                    st.write(f"Impossible de charger l'image {image_path}")
        else:
            st.write("Aucun visage similaire trouvé dans la base de données.")
    else:
        st.write("Erreur dans la recherche des visages similaires.")
