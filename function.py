import os
import io
import base64
import numpy as np
import cv2

def get_image_list(image_dir):
    """
    Retourne la liste triée des noms de fichiers d'image présents dans image_dir.
    """
    return sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

def get_mask_filename(image_filename):
    """
    Convertit le nom de fichier de l'image en nom de fichier du masque.
    Exemple:
      'berlin_000000_000019_leftImg8bit.png' ->
      'berlin_000000_000019_remapped.png'
    """
    return image_filename.replace("leftImg8bit", "remapped")

def load_and_prepare_image(image_path, target_size=(256, 256)):
    """
    Charge une image, la redimensionne et renvoie :
      - l'image redimensionnée (format uint8) pour affichage,
      - l'image normalisée (float32, valeurs [0, 1]) pour la prédiction.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Impossible de charger l'image : {image_path}")
    image = cv2.resize(image, target_size)
    image_norm = image.astype(np.float32) / 255.0
    return image, image_norm

def load_mask(mask_path, target_size=(256, 256)):
    """
    Charge un masque en niveaux de gris et le redimensionne.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Impossible de charger le masque : {mask_path}")
    mask = cv2.resize(mask, target_size)
    return mask

def predict_mask(model, image_norm):
    """
    Réalise la prédiction du masque à partir d'une image normalisée.
    La sortie du modèle est transformée en masque 2D (par argmax sur les classes).
    """
    input_image = np.expand_dims(image_norm, axis=0)  # Ajouter la dimension batch
    prediction = model.predict(input_image)[0]
    predicted_mask = np.argmax(prediction, axis=-1).astype(np.uint8)
    return predicted_mask

def colorize_mask_with_palette(mask):
    """
    Colorise un masque en niveaux de gris dont les valeurs correspondent aux classes (0 à 7)
    en utilisant une palette prédéfinie pour 8 classes.
    """
    palette = [
        (0, 0, 0),        # 0 - Void (Noir)
        (255, 0, 0),      # 1 - Human (Rouge)
        (0, 255, 0),      # 2 - Object (Vert)
        (0, 0, 255),      # 3 - Sky (Bleu)
        (255, 255, 0),    # 4 - Vehicle (Jaune)
        (255, 165, 0),    # 5 - Nature (Orange)
        (128, 0, 128),    # 6 - Construction (Violet)
        (0, 255, 255)     # 7 - Flat (Cyan)
    ]
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label in range(8):
        colored_mask[mask == label] = palette[label]
    return colored_mask

def image_to_base64(image_array, color_conversion=False, fmt='.png'):
    """
    Convertit une image (tableau numpy) en chaîne base64.
    Si color_conversion est True, convertit d'abord BGR en RGB.
    """
    if color_conversion:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    success, buffer = cv2.imencode(fmt, image_array)
    if not success:
        raise ValueError("L'encodage de l'image a échoué")
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    return img_b64
