import cv2
import mediapipe as mp
from vec1 import calculate_angle

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# TODO: load an image
Szobo = cv2.imread('C:/Users/maigu/BioMech_test/MathToCode/szoboszlai.png')

# TODO: convert the image from BGR to RGB
Szobo_conv = cv2.cvtColor(Szobo, cv2.COLOR_BGR2RGB)

# TODO: initialise the pose model

with mp_pose.Pose(static_image_mode=True) as pose:

    # TODO: run pose.process(rgb_image) on your image
    results = pose.process(Szobo_conv)
    
landmarks = results.pose_landmarks.landmark
# TODO: store landmark 23's x and y in a variable called left_hip
# hint: you access a landmark's x with landmarks[23].x
left_hip = (landmarks[23].x, landmarks[23].y)
# TODO: do the same for landmark 25 (left_knee)
left_knee = (landmarks[25].x, landmarks[25].y)
# TODO: do the same for landmark 27 (left_ankle)
left_ankle = (landmarks[27].x, landmarks[27].y)
# TODO: print all three neatly
# hint: use an f-string

print(f"left hip {left_hip}")
print(f"left knee {left_knee}")
print(f"left ankle {left_ankle}")

Szobo_angle = calculate_angle(left_hip, left_knee, left_ankle)


# TODO: convert Szobo_conv back to BGR for display
Szobo_image = cv2.cvtColor(Szobo_conv, cv2.COLOR_RGB2BGR)

# TODO: draw the landmarks on the image using mp_drawing.draw_landmarks()
# hint: it takes three arguments — the image, the landmarks, and the connections
# the connections are: mp_pose.POSE_CONNECTIONS
mp_drawing.draw_landmarks(Szobo_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

SZ_final = cv2.imwrite('SZ.png', Szobo_image)