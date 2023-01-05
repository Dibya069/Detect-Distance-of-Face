import cv2
import winsound
# import matplotlib.pyplot as plt

trained_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
known_distance = 30
known_width = 14.3
fonts = cv2.FONT_HERSHEY_COMPLEX

#focal Length finder function
def focalLength(measured_distance, real_width, width_in_ref_img):
    focal_lenth = (width_in_ref_img * measured_distance) / real_width
    return focal_lenth

#distance estimation function
def distance_finder(focal_length, real_face_width, face_width_in_frame):
    distance = (real_face_width * focal_length) / face_width_in_frame
    return distance


def face_data(image):
    face_width = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    faces = trained_face.detectMultiScale(gray_image, 1.3, 5)

    for (x, y, w, h) in faces:
        f = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        face_width = w
    return face_width


"""reading the reference image from dictionary"""
ref_img = cv2.imread("dibya.png")
ref_img_face_width = face_data(ref_img)
focal_length_found = focalLength(known_distance, known_width, ref_img_face_width)
print(focal_length_found)

cap = cv2.VideoCapture(0)

while (1):
    _, frame = cap.read()

    #calling face_data function
    facce_width_in_frame = face_data(frame)


    if facce_width_in_frame != 0:
        #finding the distance by calling the function distance finder
        distance = distance_finder(focal_length_found, known_width, facce_width_in_frame)

        #drawing text on the screen
        cv2.putText(frame, f"distance = {distance}", (50, 50), fonts, 0.6, (0, 0, 255), 2)


        #give a condition
        if distance <= 23:
            cv2.putText(frame, f"STOP you are very close", (50, 100), fonts, 0.6, (0, 0, 255), 2)
            winsound.Beep(1000, 200)
            #winsound.SND_ASYNC() #for stopping the sound and program


    cv2.imshow("Tarinee", frame)
    if cv2.waitKey(1) == ord("q"):
        break
