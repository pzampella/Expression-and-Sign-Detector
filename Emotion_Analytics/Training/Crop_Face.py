import dlib
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt

def detect_faces(image):

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames

# Load image
img_path = '/home/pzampella/Escritorio/indice.jpeg'
image = io.imread(img_path)

# Detect faces
detected_faces = detect_faces(image)

# Crop faces and plot
for n, face_rect in enumerate(detected_faces):
    print(face_rect)
    print(detected_faces[0])
    face = Image.fromarray(image).crop(face_rect)
    plt.imshow(face)
    plt.show()