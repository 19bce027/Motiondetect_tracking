import cv2, time, pandas
from datetime import datetime
import imutils
# from rembg.bg import remove
from brightness import increase_brightness

WINDOW_NAME = "window"
static_back = None

motion_list = [ None, None ]
time = []

df = pandas.DataFrame(columns = ["Start", "End"])

# Capturing video
video = cv2.VideoCapture("test1.mp4")
width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(height)
print(width)
count = 1
try :
    while True:
        count = count + 1
        # fps = video.get(cv2.CAP_PROP_FPS)
        # print(fps)
        # print("Motion list\n",motion_list)
        # print("time\n",time)
        check, frame = video.read()
        motion = 0
        frame = imutils.resize(frame, height=int(width),width=int(height))
        frame = increase_brightness(frame,60)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if static_back is None:
            static_back = gray
            continue

        diff_frame = cv2.absdiff(static_back, gray)
        thresh_frame = cv2.threshold(diff_frame, 20, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)
        cnts,_ = cv2.findContours(thresh_frame.copy(),
                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in cnts:
            if cv2.contourArea(contour) < 10000:
                continue
            motion = 1

            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        motion_list.append(motion)
        motion_list = motion_list[-2:]

        # Appending Start time of motion
        if motion_list[-1] == 1 and motion_list[-2] == 0:
            time.append(datetime.now())
        # Appending End time of motion
        if motion_list[-1] == 0 and motion_list[-2] == 1:
            time.append(datetime.now())


        cv2.imshow("Difference Frame", diff_frame)
        cv2.imshow("Color Frame", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            if motion == 1:
                time.append(datetime.now())
            break

except Exception as e :
    print(e)

print(count)
try :
    for i in range(0, len(time), 2):
        df = df.append({"Start":time[i], "End":time[i + 1]}, ignore_index = True)
except Exception as e:
    print(e)


df.to_csv("Time_of_movements.csv")
video.release()

# Destroying all the windows
cv2.destroyAllWindows()
