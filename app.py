# Importamos las librerias
from deepface import DeepFace
import cv2
import mediapipe as mp

# Inicializamos mediapipe face detection
detros = mp.solutions.face_detection
rostros = detros.FaceDetection(min_detection_confidence=0.8, model_selection=0)
dibujorostro = mp.solutions.drawing_utils

# Intentamos abrir la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

# Leemos imagen de superposición
img_overlay = cv2.imread("img.png")
if img_overlay is not None:
    img_overlay = cv2.resize(img_overlay, (0, 0), fx=0.18, fy=0.18)
    ani, ali, _ = img_overlay.shape
else:
    print("No se encontró img.png, se omite la superposición")
    img_overlay = None

# Empezamos el bucle principal
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("No se pudo capturar un frame. Revisa la cámara.")
        break

    # Convertimos a RGB para DeepFace y mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesamos detección de rostros con mediapipe
    resrostros = rostros.process(rgb)

    # Si se detectan rostros
    if resrostros.detections is not None:
        for rostro in resrostros.detections:
            # Bounding box
            al, an, _ = frame.shape
            box = rostro.location_data.relative_bounding_box
            xi, yi = int(box.xmin * an), int(box.ymin * al)
            w, h = int(box.width * an), int(box.height * al)
            xf, yf = xi + w, yi + h

            # Dibujamos rectángulo
            cv2.rectangle(frame, (xi, yi), (xf, yf), (255, 255, 0), 2)

            # Superposición de imagen
            if img_overlay is not None:
                frame[10:ani + 10, 10:ali + 10] = img_overlay

            # Análisis con DeepFace
            info = DeepFace.analyze(rgb, actions=['age', 'gender', 'emotion'], enforce_detection=False)

            # El resultado es una lista de diccionarios, tomamos el primero
            info = info[0]

            edad = info['age']
            emociones = info['dominant_emotion']
            gen = info['gender']


            # Traducciones
            traducciones = {
                'Man': 'Hombre',
                'Woman': 'Mujer',
                'angry': 'enojado/enojada',
                'disgust': 'disgustado/disgustada',
                'fear': 'miedoso/miedosa',
                'happy': 'feliz',
                'sad': 'triste',
                'surprise': 'sorprendido/sorprendida',
                'neutral': 'neutral'
            }

            # Revisa que gen sea un string
            if isinstance(gen, dict):
                gen = gen.get('gender', 'desconocido')

            # Después aplica las traducciones
            gen = traducciones.get(gen, gen)


            # Mostramos la información
            cv2.putText(frame, f"{gen}", (65, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, f"{edad}", (75, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, f"{emociones}", (75, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Mostramos el frame
    cv2.imshow("Detección de Edad y Características", frame)

    # Salimos con la tecla Esc
    if cv2.waitKey(5) == 27:
        break

# Cerramos recursos
cap.release()
cv2.destroyAllWindows()