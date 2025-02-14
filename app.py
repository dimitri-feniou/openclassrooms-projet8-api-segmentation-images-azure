import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify, render_template
import cv2
from werkzeug.utils import secure_filename

# Importation des fonctions depuis function.py
from function import (
    get_image_list,
    get_mask_filename,
    load_and_prepare_image,
    load_mask,
    predict_mask,
    colorize_mask_with_palette,
    image_to_base64
)

# Désactiver l'utilisation du GPU pour forcer le CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Charger le modèle U-Net (ajustez le chemin si besoin)
MODEL_PATH = "model/unet_model.h5"
model = keras.models.load_model(MODEL_PATH, compile=False)

# Définition de l'application Flask
app = Flask(__name__)

# Dossiers de données
IMAGE_DIR = "data/images"         # Images d'entrée
MASK_DIR = "data/masks"           # Masques réels associés
PREDICTION_DIR = "data/predictions"  # Pour sauvegarder les résultats
os.makedirs(PREDICTION_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

@app.route("/images", methods=["GET"])
def list_images():
    images = get_image_list(IMAGE_DIR)
    return jsonify({"images": images})

@app.route("/predict/<image_id>", methods=["GET"])
def predict(image_id):
    image_path = os.path.join(IMAGE_DIR, image_id)
    # Conversion du nom de fichier de l'image en nom de fichier du masque réel
    mask_filename = get_mask_filename(image_id)
    mask_path = os.path.join(MASK_DIR, mask_filename)

    # Vérifier que l'image existe
    if not os.path.exists(image_path):
        return jsonify({"error": "Image non trouvée"}), 404

    try:
        # Chargement de l'image et préparation
        image_orig, image_norm = load_and_prepare_image(image_path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 500

    # Prédiction du masque
    predicted_mask = predict_mask(model, image_norm)
    # Colorisation du masque prédit
    predicted_mask_color = colorize_mask_with_palette(predicted_mask)
    predicted_mask_b64 = image_to_base64(predicted_mask_color)

    # Gestion du masque réel
    if os.path.exists(mask_path):
        try:
            real_mask = load_mask(mask_path)
            real_mask_color = colorize_mask_with_palette(real_mask)
            mask_real_b64 = image_to_base64(real_mask_color)
            # Enregistrer le masque réel colorisé
            real_mask_save_path = os.path.join(PREDICTION_DIR, f"real_mask_{image_id}")
            cv2.imwrite(real_mask_save_path, real_mask_color)
        except ValueError as e:
            print(f"Erreur lors du chargement du masque réel : {e}")
            mask_real_b64 = None
            real_mask_save_path = None
    else:
        print(f"Le masque réel n'existe pas pour l'image {image_id}")
        mask_real_b64 = None
        real_mask_save_path = None

    # Conversion de l'image d'origine en RGB pour l'affichage
    image_orig_rgb = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
    image_orig_b64 = image_to_base64(image_orig_rgb, color_conversion=False)

    # Sauvegarde du masque prédit en couleur dans le dossier PREDICTION_DIR
    predicted_mask_save_path = os.path.join(PREDICTION_DIR, f"predicted_mask_{image_id}")
    cv2.imwrite(predicted_mask_save_path, predicted_mask_color)

    return jsonify({
        "message": "Prédiction effectuée",
        "image_id": image_id,
        "image": image_orig_b64,
        "mask_ground_truth": mask_real_b64,
        "mask_predicted": predicted_mask_b64,
        "predicted_mask_file": predicted_mask_save_path,
        "real_mask_file": real_mask_save_path
    })

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        # Récupérer les fichiers envoyés via le formulaire
        image_file = request.files.get("image")
        mask_file = request.files.get("mask")
        messages = {}

        if image_file:
            # Sécuriser le nom du fichier et enregistrer dans IMAGE_DIR
            image_filename = secure_filename(image_file.filename)
            image_path = os.path.join(IMAGE_DIR, image_filename)
            image_file.save(image_path)
            messages["image"] = f"Image uploaded as {image_filename}"
        else:
            messages["image"] = "No image file uploaded"

        if mask_file:
            mask_filename = secure_filename(mask_file.filename)
            mask_path = os.path.join(MASK_DIR, mask_filename)
            mask_file.save(mask_path)
            messages["mask"] = f"Mask uploaded as {mask_filename}"
        else:
            messages["mask"] = "No mask file uploaded"

        return jsonify(messages)
    else:
        # Pour GET, afficher un formulaire d'upload
        return render_template("upload.html")

@app.route("/", methods=["GET", "POST"])
def index():
    image_list = get_image_list(IMAGE_DIR)
    results = None
    selected_image = None

    if request.method == "POST":
        selected_image = request.form.get("image_id")
        if selected_image:
            response = predict(selected_image)
            if response.status_code == 200:
                results = response.get_json()
            else:
                results = {"error": "Erreur lors de la prédiction"}
    return render_template("index.html", image_list=image_list, results=results, selected_image=selected_image)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)
