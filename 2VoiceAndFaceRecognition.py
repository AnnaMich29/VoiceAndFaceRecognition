import cv2
import pyttsx3
import speech_recognition
import pyaudio
from datetime import datetime
video_capture = cv2.VideoCapture(0)
face_classifier =  cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
r=speech_recognition.Recognizer()
def SpeakText(command):
    engine=pyttsx3.init()
    rate=engine.getProperty('rate')
    engine.setProperty('rate',rate-80)
    engine.say(command)
    engine.runAndWait()

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x+w, y+h), (0, 255, 0), 4)


def Listen():
    with speech_recognition.Microphone() as source2:
        r.adjust_for_ambient_noise(source2, duration=1)
        SpeakText("say something")
        audio2=r.listen(source2)
        try:
            MyText = r.recognize_google(audio2)
        except:
            SpeakText("Didnt recognize voice")
            SpeakText("try again")
            Listen()
            return
        SpeakText("You said" + MyText)
        if MyText == "open camera":
            while True:
                result, video_frame = video_capture.read()
                if result is False:
                    break
                faces = detect_bounding_box(
                    video_frame
                )
                now = datetime.now()
                if type(faces) is not tuple:
                    img_name = "frame_at_time" + str(now.hour) + "." + str(now.minute) + "." + str(now.second) + ".png"
                    cv2.imwrite(img_name, video_frame)
                cv2.imshow(
                    "My Face Detection Project", video_frame
                )
                k = cv2.waitKey(1)
                if k % 256 == 27:
                    # ESC pressed
                    print("Escape hit, closing...")
                    break
            video_capture.release()
            cv2.destroyAllWindows()
Listen()
