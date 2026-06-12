import cv2 # type: ignore
import mediapipe as mp # type: ignore
from vec1 import calculate_angle
import numpy as np
import matplotlib.pyplot as plt


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


L_Angles = []
R_Angles = []

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
            R_Hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            R_Knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
            R_Ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
            
                       
            L_Hip_arr = np.array([L_Hip.x * fr_width, L_Hip.y * fr_height])
            L_Knee_arr = np.array([L_Knee.x * fr_width, L_Knee.y * fr_height])
            L_Ankle_arr = np.array([L_Ankle.x * fr_width, L_Ankle.y * fr_height])
            R_Hip_arr = np.array([R_Hip.x * fr_width, R_Hip.y * fr_height])
            R_Knee_arr = np.array([R_Knee.x * fr_width, R_Knee.y * fr_height])
            R_Ankle_arr = np.array([R_Ankle.x * fr_width, R_Ankle.y * fr_height])
            
            angle = calculate_angle(L_Hip_arr, L_Knee_arr, L_Ankle_arr)
            R_angle = calculate_angle(R_Hip_arr, R_Knee_arr, R_Ankle_arr)
            
            L_Angles.append(round(angle, 1))
            R_Angles.append(round(R_angle, 1))
            
            cv2.putText(frame, f"Left: {round(angle, 1)}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Right: {round(R_angle, 1)}", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
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

L_Avg = np.average(L_Angles)
R_Avg = np.average(R_Angles)

deepest_L = min(L_Angles)
straightest_L = max(L_Angles)

deepest_R = min(R_Angles)
straightest_R = max(R_Angles)

print(f"Left Knee | Avg: {round(L_Avg, 2)} | Deepest: {round(deepest_L, 2)} | Straightest: {round(straightest_L, 2)}")
print(f"Right Knee | Avg: {round(R_Avg, 2)} | Deepest: {round(deepest_R, 2)} | Straightest: {round(straightest_R, 2)}")

plt.plot(L_Angles, label="Left Knee")
plt.plot(R_Angles, label="Right Knee")

plt.title("Right vs Left knee")
plt.xlabel("Frames")
plt.ylabel("Angle")
plt.legend()
plt.savefig("Knee_angles.png", dpi=300, bbox_inches ="tight")
plt.show()