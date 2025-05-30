import cv2  
import mediapipe as mp
import numpy as np

# Configuración de YOLO
def setup_yolo():
    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Iniciar cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Cargar YOLO
yolo_net, output_layers = setup_yolo()

# Diccionario de emociones con umbrales y colores
EMOTIONS = {
    "feliz": {
        "threshold": 0.03,
        "color": (0, 255, 0)  # Verde
    },
    "sorprendido": {
        "threshold": 0.015,
        "color": (255, 255, 0)  # Amarillo
    },
    "enojado": {
        "threshold": -0.01,
        "color": (0, 0, 255)  # Rojo
    },
    "triste": {
        "threshold": 0.02,
        "color": (255, 0, 0)  # Azul
    },
    "neutral": {
        "threshold": 0.01,
        "color": (255, 255, 255)  # Blanco
    }
}

def detect_emotion(landmarks):
    # Puntos clave para análisis de emociones
    # Labios
    upper_lip = landmarks[13].y
    lower_lip = landmarks[14].y
    lip_distance = abs(upper_lip - lower_lip)
    
    # Cejas
    left_eyebrow = landmarks[159].y
    right_eyebrow = landmarks[386].y
    eyebrow_diff = abs(left_eyebrow - right_eyebrow)
    
    # Ojos
    left_eye = landmarks[145].y - landmarks[159].y
    right_eye = landmarks[374].y - landmarks[386].y
    eye_openness = (left_eye + right_eye) / 2
    
    # Frente
    forehead = landmarks[10].y
    nose_tip = landmarks[4].y
    vertical_ratio = nose_tip - forehead
    
    # Determinar emoción basada en los umbrales
    if lip_distance > EMOTIONS["feliz"]["threshold"]:
        return "feliz"
    elif eyebrow_diff > EMOTIONS["sorprendido"]["threshold"]:
        return "sorprendido"
    elif vertical_ratio < EMOTIONS["enojado"]["threshold"]:
        return "enojado"
    elif eye_openness < EMOTIONS["triste"]["threshold"]:
        return "triste"
    else:
        return "neutral"

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error al capturar frame")
            continue

        height, width = frame.shape[:2]
        
        # Detección con YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        yolo_net.setInput(blob)
        outs = yolo_net.forward(output_layers)
        
        # Procesar detecciones
        boxes = []
        confidences = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5 and class_id == 0:  # Solo personas
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
        
        # Eliminar detecciones solapadas
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # Procesar cada rostro detectado
        for i in indices:
            x, y, w, h = boxes[i]
            
            # Dibujar bounding box (rojo)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Recortar región del rostro
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size == 0:
                continue
                
            # Convertir a RGB para MediaPipe
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            
            # Análisis con MediaPipe
            results = face_mesh.process(face_rgb)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Convertir landmarks a coordenadas
                    landmarks = []
                    for landmark in face_landmarks.landmark:
                        lx = x + int(landmark.x * w)
                        ly = y + int(landmark.y * h)
                        landmarks.append(mp.solutions.face_mesh.NormalizedLandmark(x=landmark.x, y=landmark.y))
                    
                    # Detectar emoción
                    emotion = detect_emotion(landmarks)
                    emotion_color = EMOTIONS[emotion]["color"]
                    
                    # Dibujar contorno facial del color de la emoción
                    face_outline = [10, 338, 297, 332, 284, 251, 389, 356, 454, 
                                  323, 361, 288, 397, 365, 379, 378, 400, 377, 
                                  152, 148, 176, 149, 150, 136, 172, 58, 132, 
                                  93, 234, 127, 162, 21, 54, 103, 67, 109]
                    
                    points = []
                    for idx in face_outline:
                        point = (x + int(landmarks[idx].x * w), y + int(landmarks[idx].y * h))
                        points.append(point)
                    
                    # Dibujar polígono del rostro
                    cv2.polylines(frame, [np.array(points)], True, emotion_color, 2)
                    
                    # Mostrar información
                    cv2.putText(frame, f"Emocion: {emotion}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)
                    
                    # Dibujar puntos clave de expresiones
                    keypoints = {
                        "labios": [13, 14],
                        "cejas": [159, 386],
                        "ojos": [145, 374]
                    }
                    
                    for group, indices in keypoints.items():
                        for idx in indices:
                            point = (x + int(landmarks[idx].x * w), y + int(landmarks[idx].y * h))
                            cv2.circle(frame, point, 3, emotion_color, -1)

        # Mostrar frame
        cv2.imshow('Deteccion Facial con Analisis de Emociones', frame)
        
        # Salir con ESC
        if cv2.waitKey(5) == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()