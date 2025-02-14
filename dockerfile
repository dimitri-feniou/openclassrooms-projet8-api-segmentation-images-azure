FROM python:3.10-slim

# Mettre à jour les dépôts et installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier le fichier requirements.txt et installer les dépendances Python
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Créer la structure de dossiers pour les données
RUN mkdir -p data/images data/masks data/predictions

# Copier l'ensemble du code de l'application dans le conteneur
COPY . .

# Exposer le port (par défaut Flask utilise le port 5000)
EXPOSE 80

# Lancer l'application
CMD ["python", "app.py"]
