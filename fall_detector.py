"""
Fall Detection System - OpenCV + MediaPipe Pose
Detecta caídas de personas en tiempo real usando cámara.

Cómo funciona:
1. Detecta el esqueleto de la persona (33 puntos del cuerpo) con MediaPipe.
2. Calcula el ángulo del torso (hombros -> caderas) respecto a la vertical.
3. Si el cuerpo pasa de vertical a horizontal y se mantiene -> ALERTA DE CAÍDA.
"""

import cv2
import mediapipe as mp
import time
import math
import boto3

# --- Configuración AWS SNS ---
AWS_REGION = "eu-north-1"
SNS_TOPIC_ARN = "arn:aws:sns:eu-north-1:185489190419:fall-alerts"

sns_client = boto3.client("sns", region_name=AWS_REGION)
ALERT_COOLDOWN_SECONDS = 30
last_alert_time = 0


def send_fall_alert():
    global last_alert_time
    now = time.time()
    if now - last_alert_time < ALERT_COOLDOWN_SECONDS:
        return
    try:
        sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Message="ALERTA: se ha detectado una posible caida. Revisa la camara inmediatamente.",
            Subject="Alerta de caida detectada"
        )
        print("[AWS SNS] Alerta enviada correctamente.")
        last_alert_time = now
    except Exception as e:
        print(f"[AWS SNS] Error al enviar alerta: {e}")


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

ANGLE_THRESHOLD = 45
FALL_HOLD_SECONDS = 3


def calculate_torso_angle(landmarks, w, h):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    shoulder_mid = ((left_shoulder.x + right_shoulder.x) / 2 * w,
                     (left_shoulder.y + right_shoulder.y) / 2 * h)
    hip_mid = ((left_hip.x + right_hip.x) / 2 * w,
               (left_hip.y + right_hip.y) / 2 * h)

    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]

    angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))
    return angle


def main(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("No se pudo abrir la camara/fuente de video.")
        return

    fall_start_time = None
    fall_confirmed = False

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            status_text = "Sin persona detectada"
            status_color = (200, 200, 200)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

                angle = calculate_torso_angle(results.pose_landmarks.landmark, w, h)

                if angle > ANGLE_THRESHOLD:
                    if fall_start_time is None:
                        fall_start_time = time.time()
                    elapsed = time.time() - fall_start_time
                    if elapsed >= FALL_HOLD_SECONDS:
                        if not fall_confirmed:
                            send_fall_alert()
                        fall_confirmed = True
                        status_text = "ALERTA: CAIDA DETECTADA"
                        status_color = (0, 0, 255)
                    else:
                        status_text = f"Postura anomala ({elapsed:.1f}s)"
                        status_color = (0, 165, 255)
                else:
                    fall_start_time = None
                    fall_confirmed = False
                    status_text = "Persona de pie / normal"
                    status_color = (0, 200, 0)

                cv2.putText(frame, f"Angulo torso: {angle:.1f} grados", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            if fall_confirmed:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)

            cv2.imshow("Fall Detection System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(0)


