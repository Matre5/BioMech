import cv2 # type: ignore
import mediapipe as mp # type: ignore
from vec1 import calculate_angle
import numpy as np


mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False)

vid = cv2.VideoCapture(r'C:\Users\maigu\BioMech_test\MathToCode\sport_clip.mp4')
print(vid.isOpened())

fps = vid.get(cv2.CAP_PROP_FPS)
fr_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
fr_width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
fr_height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(fps)
print(fr_count)
print(fr_width)
print(fr_height)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (int(fr_width), int(fr_height)) )


while True:
    ret, frame = vid.read()

    
    if not ret:
        break
    else:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            
            L_Hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            L_Knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
            L_Ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
            
            L_Hip_arr = np.array([L_Hip.x * fr_width, L_Hip.y * fr_height])
            L_Knee_arr = np.array([L_Knee.x * fr_width, L_Knee.y * fr_height])
            L_Ankle_arr = np.array([L_Ankle.x * fr_width, L_Ankle.y * fr_height])
            
            angle = calculate_angle(L_Hip_arr, L_Knee_arr, L_Ankle_arr)
            
            cv2.putText(frame, str(round(angle, 1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
            # print(angle)
            
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
        out.write(frame)
        cv2.imshow('Frames picture', frame)
        cv2.waitKey(int(1000/fps))

out.release()
vid.release()