import cv2
import mediapipe as mp
import math
import screen_brightness_control as sbc

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            thumb = hand.landmark[4]
            index = hand.landmark[8]

            h, w, _ = frame.shape

            tx = int(thumb.x * w)
            ty = int(thumb.y * h)

            ix = int(index.x * w)
            iy = int(index.y * h)

            # Distance
            distance = math.hypot(ix - tx, iy - ty)

            # Brightness 0-100
            brightness = max(
                0,
                min(100, int((distance - 20) / 150 * 100))
            )

            sbc.set_brightness(brightness)

            cv2.putText(
                frame,
                f"Brightness: {brightness}%",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Virtual Brightness Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
