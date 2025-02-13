import unittest
import json
from app import app

class APITestCase(unittest.TestCase):
    def setUp(self):
        # Configurer l'application pour le mode test
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_list_images(self):
        """Vérifie que l'endpoint /images retourne une liste d'images."""
        response = self.client.get('/images')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode())
        self.assertIn('images', data)
        self.assertIsInstance(data['images'], list)

    def test_predict_invalid_image(self):
        """Vérifie que l'endpoint /predict pour une image inexistante renvoie une erreur 404."""
        response = self.client.get('/predict/non_existent_image.png')
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data.decode())
        self.assertIn('error', data)

    def test_predict_valid_image(self):
        """
        Vérifie l'endpoint /predict pour une image existante.
        On récupère d'abord la liste des images et on teste avec la première image.
        """
        images_response = self.client.get('/images')
        images_data = json.loads(images_response.data.decode())
        images_list = images_data.get('images', [])
        if not images_list:
            self.skipTest("Aucune image disponible dans IMAGE_DIR pour le test.")
        image_id = images_list[0]
        response = self.client.get(f'/predict/{image_id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode())
        # Vérifier que certains éléments clés sont présents dans la réponse
        self.assertIn('message', data)
        self.assertIn('image_id', data)
        self.assertIn('image', data)
        self.assertIn('mask_predicted', data)
        # "mask_ground_truth" peut être None si le masque n'existe pas

    def test_index_page(self):
        """Vérifie que la page d'accueil (interface web) est bien accessible et renvoie du HTML."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<html', response.data)

if __name__ == '__main__':
    unittest.main()
