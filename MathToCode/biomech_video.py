import cv2 # type: ignore
import mediapipe as mp # type: ignore


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

while True:
    ret, frame = vid.read()

    
    if not ret:
        break
    else:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
        cv2.imshow('Frames picture', frame)
        cv2.waitKey(int(1000/fps))

vid.release()