import cv2
import time
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
cap = cv2.VideoCapture(0)

phone_start_time = None
last_drink_time = time.time()
person_missing_time = None

PHONE_THRESHOLD = 2        # seconds
HYDRATION_LIMIT = 30      # demo (30 sec instead of 30 min)
EMPTY_DESK_LIMIT = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    person_detected = False
    phone_detected = False
    cup_detected = False
    bottle_detected = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            frame_color = (0, 255, 0)

            if cls == 0:
                person_detected = True
                frame_color = (0, 255, 0)
                # label = 'PERSON'

            if cls == 67:
                phone_detected = True
                frame_color = (0, 0, 255)
                # label = 'PHONE'

            if cls == 41:
                cup_detected = True
                frame_color = (255,0,0)
                # label = 'CUP'

            if cls == 39:
                bottle_detected = True
                frame_color = (255,0,0)
                # label = 'BOTTLE'

            cv2.rectangle(frame,(x1,y1),(x2,y2),frame_color,1)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, frame_color, 2)

    current_time = time.time()

    # HYDRATION RESET
    if cup_detected or bottle_detected:
        last_drink_time = current_time

    # EMPTY DESK
    if not person_detected:
        if person_missing_time is None:
            person_missing_time = current_time

        if current_time - person_missing_time > EMPTY_DESK_LIMIT:
            cv2.putText(frame,"System Paused: User Away",
                        (30,40),cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,(0,255,255),2)
    else:
        person_missing_time = None

    # PHONE DISTRACTION
    # if phone_detected:
    #     if phone_start_time is None:
    #         phone_start_time = current_time

    #     if current_time - phone_start_time > PHONE_THRESHOLD:
    #         cv2.putText(frame,"WARNING: PUT PHONE AWAY",
    #                     (30,80),cv2.FONT_HERSHEY_SIMPLEX,
    #                     0.5,(0,0,255),2)
    # else:
    #     phone_start_time = None

    # HYDRATION ALERT
    if current_time - last_drink_time > HYDRATION_LIMIT:
        cv2.putText(frame,"HEALTH ALERT: Drink Water",
                    (30,120),cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,(255,0,0),2)

    # DEEP WORK
    if person_detected:
        if not phone_detected:
            cv2.putText(frame,"Status: Focusing",
                        (30,160),cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,(0,255,0),2)
        elif phone_detected:
            if phone_start_time is None:
                phone_start_time = current_time

            if current_time - phone_start_time > PHONE_THRESHOLD:
                cv2.putText(frame,"WARNING: PUT PHONE AWAY",
                            (30,80),cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,(0,0,255),2)
        else:
            phone_start_time = None

    cv2.imshow("AuraGuard Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()