import io
import json
import sys
import threading
import unittest
import urllib.request
import urllib.error
from pathlib import Path
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from inference import read_image, ordered_probabilities, catalog, predict
from server import Handler


def image_bytes(size=(400, 200)):
    data = io.BytesIO()
    Image.new('RGB', size, 'red').save(data, 'PNG')
    return data.getvalue()


class PreprocessingTests(unittest.TestCase):
    def test_landscape_padding(self):
        image = read_image(image_bytes())
        self.assertEqual(image.size, (256, 256))
        self.assertEqual(image.getpixel((128, 63)), (0, 0, 0))
        self.assertEqual(image.getpixel((128, 64)), (255, 0, 0))
        self.assertEqual(image.getpixel((128, 191)), (255, 0, 0))
        self.assertEqual(image.getpixel((128, 192)), (0, 0, 0))

    def test_portrait_and_small(self):
        for size in [(200, 400), (10, 20)]:
            image = read_image(image_bytes(size))
            self.assertEqual(image.getpixel((63, 128)), (0, 0, 0))
            self.assertEqual(image.getpixel((64, 128)), (255, 0, 0))

    def test_exif_rotation(self):
        source = Image.new('RGB', (400, 200), 'red')
        exif = source.getexif(); exif[274] = 6
        data = io.BytesIO(); source.save(data, 'JPEG', exif=exif)
        image = read_image(data.getvalue())
        self.assertEqual(image.getpixel((10, 128)), (0, 0, 0))
        self.assertGreater(image.getpixel((128, 10))[0], 240)

    def test_label_mapping(self):
        result = ordered_probabilities([.1, .2, .3, .4], ['Green', 'Overripe', 'Ripe', 'Semi-ripe'])
        self.assertEqual([r['probability'] for r in result], [.1, .4, .3, .2])

    def test_invalid_output(self):
        for values in ([0, 0, 0, 0], [float('nan'), 0, 0, 1], [-1, 0, 1, 1]):
            with self.assertRaises(ValueError):
                ordered_probabilities(values, ['Green', 'Overripe', 'Ripe', 'Semi-ripe'])

    def test_unknown_model(self):
        with self.assertRaises(ValueError):
            predict(read_image(image_bytes()), 'unknown')


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()

    def test_preview(self):
        with urllib.request.urlopen(self.base + '/api/preview', data=image_bytes()) as response:
            result = json.load(response)
        self.assertEqual((result['width'], result['height']), (256, 256))
        self.assertTrue(result['preview'].startswith('data:image/png;base64,'))

    def test_invalid_image(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(self.base + '/api/preview', data=b'not an image')
        self.assertEqual(error.exception.code, 400)

    def test_missing_model(self):
        missing = next((m for m in catalog() if not m['available']), None)
        if missing is None:
            self.skipTest('All checkpoints installed')
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(self.base + '/api/predict?model=' + missing['id'], data=image_bytes())
        self.assertEqual(error.exception.code, 409)

    def test_catalog_and_page(self):
        with urllib.request.urlopen(self.base + '/api/models') as response:
            self.assertEqual(len(json.load(response)['models']), 9)
        with urllib.request.urlopen(self.base + '/') as response:
            self.assertIn(b'BANANA LAB', response.read())


if __name__ == '__main__':
    unittest.main()
