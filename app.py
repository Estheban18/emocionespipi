import cv2
import numpy as np
import time
import os

class AdvancedEmotionDetector:
    def __init__(self):
        """Inicializa el detector de emociones con configuraciones avanzadas."""
        # Configuración inicial
        self.MIN_CONFIDENCE = 0.5
        self.USE_GPU = self._check_gpu_support()
        self.emotion_labels = ['Enojo', 'Disgusto', 'Miedo', 'Feliz', 'Triste', 'Sorpresa', 'Neutral']
        
        # Inicialización de modelos
        self.face_detector = self._initialize_face_detector()
        self.emotion_detector = self._initialize_emotion_detector()
        
        # Métricas de rendimiento
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.avg_processing_time = 0
        
        # Configuración de visualización
        self.colors = {
            'high_confidence': (0, 255, 0),      # Verde
            'medium_confidence': (0, 200, 255),  # Naranja
            'low_confidence': (0, 0, 255),       # Rojo
            'text': (255, 255, 255),            # Blanco
            'background': (50, 50, 50)          # Gris oscuro
        }

    def _check_gpu_support(self):
        """Verifica si hay soporte para GPU."""
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print("[INFO] GPU detectada - Activando aceleración CUDA")
            return True
        print("[INFO] No se detectó GPU - Usando CPU")
        return False

    def _initialize_face_detector(self):
        """Inicializa el detector facial usando el modelo Caffe."""
        # Ruta a los archivos del modelo (deben estar en el mismo directorio)
        model_path = "face_detection_model"
        model_file = os.path.join(model_path, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
        config_file = os.path.join(model_path, "deploy.prototxt")
        
        # Cargar red neuronal
        net = cv2.dnn.readNetFromCaffe(config_file, model_file)
        
        if self.USE_GPU:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        return net

    def _initialize_emotion_detector(self):
        """Inicializa el detector de emociones usando el modelo Caffe."""
        # Ruta a los archivos del modelo (deben estar en el mismo directorio)
        model_path = "emotion_detection_model"
        model_file = os.path.join(model_path, "emotion_net.caffemodel")
        config_file = os.path.join(model_path, "emotion_net.prototxt")
        
        # Cargar red neuronal
        net = cv2.dnn.readNetFromCaffe(config_file, model_file)
        
        if self.USE_GPU:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        return net

    def _preprocess_face(self, face_img):
        """Preprocesa la imagen del rostro para el modelo de emociones."""
        # Convertir a RGB si es necesario
        if len(face_img.shape) == 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
        
        # Crear blob para el modelo
        blob = cv2.dnn.blobFromImage(
            face_img, 
            scalefactor=1.0/127.5, 
            size=(64, 64), 
            mean=[127.5, 127.5, 127.5],
            swapRB=True,
            crop=False
        )
        return blob

    def _update_performance_metrics(self):
        """Actualiza las métricas de rendimiento."""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if elapsed >= 1.0:  # Actualizar FPS cada segundo
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = current_time

    def _get_confidence_color(self, confidence):
        """Devuelve el color según el nivel de confianza."""
        if confidence > 0.7:
            return self.colors['high_confidence']
        elif confidence > 0.5:
            return self.colors['medium_confidence']
        else:
            return self.colors['low_confidence']

    def _draw_detection_results(self, frame, face_rect, emotion, confidence):
        """Dibuja los resultados de la detección en el frame."""
        x, y, w, h = face_rect
        color = self._get_confidence_color(confidence)
        
        # Dibujar rectángulo alrededor del rostro
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Dibujar fondo para el texto
        cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)
        cv2.rectangle(frame, (x, y-30), (x+w, y), color, 2)
        
        # Mostrar emoción y confianza
        label = f"{emotion}: {confidence:.0%}"
        cv2.putText(frame, label, (x+5, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
        
        # Mostrar información de rendimiento
        perf_label = f"FPS: {self.fps:.1f} | Resolución: {frame.shape[1]}x{frame.shape[0]}"
        cv2.putText(frame, perf_label, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)

    def detect_emotions(self, frame):
        """Procesa un frame y detecta emociones."""
        start_time = time.time()
        h, w = frame.shape[:2]
        
        # Crear blob para detección facial
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), 
            [104, 117, 123], False, False
        )
        
        # Detectar rostros
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        # Procesar detecciones
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.MIN_CONFIDENCE:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                # Asegurar que las coordenadas estén dentro del frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w-1, x2), min(h-1, y2)
                
                # Extraer ROI del rostro
                face_roi = frame[y1:y2, x1:x2]
                
                if face_roi.size == 0:
                    continue
                
                # Preprocesar y detectar emoción
                emotion_blob = self._preprocess_face(face_roi)
                self.emotion_detector.setInput(emotion_blob)
                preds = self.emotion_detector.forward()
                
                # Obtener emoción predominante
                emotion_idx = np.argmax(preds)
                emotion = self.emotion_labels[emotion_idx]
                confidence = float(preds[0, emotion_idx])
                
                # Dibujar resultados
                self._draw_detection_results(frame, (x1, y1, x2-x1, y2-y1), emotion, confidence)
        
        # Actualizar métricas de rendimiento
        self._update_performance_metrics()
        self.avg_processing_time = 0.9 * self.avg_processing_time + 0.1 * (time.time() - start_time)
        
        return frame

def main():
    """Función principal para ejecutar el detector de emociones."""
    # Inicializar detector
    detector = AdvancedEmotionDetector()
    
    # Configurar captura de video
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Configurar ventana de visualización
    cv2.namedWindow("Detección de Emociones Avanzada", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detección de Emociones Avanzada", 1280, 720)
    
    print("[INFO] Iniciando detección de emociones. Presione 'ESC' para salir.")
    
    try:
        while True:
            # Leer frame
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] No se pudo capturar el frame.")
                break
            
            # Voltear frame horizontalmente para efecto espejo
            frame = cv2.flip(frame, 1)
            
            # Procesar frame
            output_frame = detector.detect_emotions(frame)
            
            # Mostrar resultado
            cv2.imshow("Detección de Emociones Avanzada", output_frame)
            
            # Salir con ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    except Exception as e:
        print(f"[ERROR] Ocurrió una excepción: {str(e)}")
    
    finally:
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Programa terminado correctamente.")

if __name__ == "__main__":
    main()