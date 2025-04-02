import cv2
import face_recognition
import numpy as np

# 已知臉部（提前準備一張照片）
known_image = face_recognition.load_image_file("隊員A.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]
known_faces = [known_encoding]
names = ["隊員A"]

# 手機串流URL（從IP Webcam或EpocCam拿到）
video_url = "http://192.168.0.185:8080"  # 換成你的URL

# 開啟串流
cap = cv2.VideoCapture(video_url)

while True:
    ret, frame = cap.read()  # 讀取一幀影像
    if not ret:
        print("無法讀取影像，檢查URL或網路")
        break

    # 轉RGB格式給face_recognition用
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 檢測和辨識臉部
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # 比對每張臉
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"
        if True in matches:
            name = names[matches.index(True)]
        
        # 在影像上畫框和名字
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 顯示結果
    cv2.imshow("Facial Recognition", frame)

    # 按'q'退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理
cap.release()
cv2.destroyAllWindows()