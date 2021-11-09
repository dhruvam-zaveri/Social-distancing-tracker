import cv2
import datetime
import imutils
import numpy as np
import math
from centroidtracker import CentroidTracker
from itertools import combinations

click = 0
points = []
perspective_kernel = []
distance_w = 0
distance_h = 0
protopath = "./model/MobileNetSSD_deploy.prototxt"
modelpath = "./model/MobileNetSSD_deploy.caffemodel"

detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
tracker = CentroidTracker(maxDisappeared=70, maxDistance=80)

# These are the classes of objects that can be detected by using the MobileNet SSD object detection algorithm
CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

def cal_dis(p1, p2):
    global distance_w, distance_h

    h = abs(p2[1] - p1[1])
    w = abs(p2[0] - p1[0])

    dis_w = float((w / distance_w) * 180)
    dis_h = float((h / distance_h) * 180)

    return int(np.sqrt(((dis_h) ** 2) + ((dis_w) ** 2)))


# To scale the bird's eye view window
def get_scale(W, H):

    dis_w = 400
    dis_h = 600

    return float(dis_w / W), float(dis_h / H)


def get_perspective_transform(H, W):
    global perspective_kernel, distance_w, distance_h
    src = np.array(points[:4], dtype=np.float32)
    dst = np.array([[0, H], [W, H], [W, 0], [0, 0]], dtype=np.float32)

    # Perspective convolution (3x3)
    perspective_kernel = cv2.getPerspectiveTransform(src, dst)

    # Adjusting the original frame according to the perspective convolution M
    perspective_transform = cv2.warpPerspective(frames, perspective_kernel, (H, W))
    # cv2.imshow("perspective transform", perspective_transform)

    pts = np.array([points[4:7]], dtype=np.float32)
    warped_pt = cv2.perspectiveTransform(pts, perspective_kernel)[0]

    # since bird eye view has property that all points are equidistant in horizontal and vertical direction.
    # distance_w and distance_h will give us 180 cm distance in both horizontal and vertical directions
    # (how many pixels will be there in 180cm length in horizontal and vertical direction of birds eye view),
    # which we can use to calculate distance between two humans in transformed view or bird eye view
    distance_w = (
        (warped_pt[0][0] - warped_pt[1][0]) ** 2
        + (warped_pt[0][1] - warped_pt[1][1]) ** 2
    ) ** 0.5
    distance_h = (
        (warped_pt[0][0] - warped_pt[2][0]) ** 2
        + (warped_pt[0][1] - warped_pt[2][1]) ** 2
    ) ** 0.5


def bird_eye_view(frames, red_mat, green_mat, scale_w, scale_h):
    h = frame.shape[0]
    w = frame.shape[1]

    red = (0, 0, 255)
    green = (0, 255, 0)
    white = (200, 200, 200)

    blank_image = np.zeros((int(h * scale_h), int(w * scale_w), 3), np.uint8)
    blank_image[:] = white
    for i in green_mat:
        blank_image = cv2.circle(
            blank_image, (int(i[0] * scale_w), int(i[1] * scale_h)), 5, green, 10
        )
    for i in red_mat:
        blank_image = cv2.circle(
            blank_image, (int(i[0] * scale_w), int(i[1] * scale_h)), 5, red, 10
        )

    return blank_image


def get_points(event, x, y, flags, params):
    global click, points

    if event == cv2.EVENT_LBUTTONDOWN and click < 4:
        click += 1
        cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)

        if len(points) == 3:
            cv2.line(
                frame,
                (x, y),
                points[-1],
                (70, 70, 70),
                2,
            )
            cv2.line(
                frame,
                (x, y),
                points[0],
                (70, 70, 70),
                2,
            )

        elif len(points) >= 1:
            cv2.line(
                frame,
                (x, y),
                points[-1],
                (70, 70, 70),
                2,
            )

        points.append((x, y))
        cv2.imshow("point selection", frame)

    elif event == cv2.EVENT_LBUTTONDOWN and click >= 4 and click < 7:
        click += 1
        points.append((x, y))

        cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
        cv2.line(frame, (x, y), (points[4][0], points[4][1]), (70, 70, 70), 2)

        cv2.imshow("point selection", frame)

    elif click >= 7:
        # calcDistance()
        cv2.destroyWindow("point selection")
        return


def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(
                idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
            )

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))


if __name__ == "__main__":
    cap = cv2.VideoCapture("./data/testvideo1.mp4")

    while True:
        ret, frame = cap.read()
        H, W = frame.shape[0], frame.shape[1]

        frame = imutils.resize(frame, width=W, height=H)
        scale_w, scale_h = get_scale(W, H)

        cv2.imshow("point selection", frame)
        cv2.setMouseCallback("point selection", get_points)
        key = cv2.waitKey(0)

        if click >= 7:
            cap.release()
            break

    cap = cv2.VideoCapture("data/testvideo1.mp4")

    # To calculate the FPS value we will the the start time and the end time and then subtract the end time from the
    # start time and dividing that by the number of frames we will get the value of the FPS

    # This will be recording the start time
    fps_start_time = datetime.datetime.now()

    # This will keep track of the fps value
    fps = 0

    # This will keep track of the total frames that are there in a video
    total_frames = 0

    # This dictionary is used to keep the values of centroid
    centroid_dict = dict()

    while True:
        ret, frames = cap.read()
        total_frames += 1
        fps_end_time = datetime.datetime.now()

        # Calculating the height and the width of the frame
        H, W = frames.shape[:2]
        get_perspective_transform(H, W)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

        # Converting the image into a blob so that we can use it for detection purpose
        blob = cv2.dnn.blobFromImage(frames, 0.007843, (W, H), 127.5)

        # Giving input to the detector
        detector.setInput(blob)

        # Collecting all the detected objects in the variable person_detections
        person_detections = detector.forward()

        # In this list we will be storing all the coordinates of the bounding box
        rects = []

        # Collecting only the person object from all the collected objects
        for i in np.arange(0, person_detections.shape[2]):

            # Accessing the confidence of the detection
            confidence = person_detections[0, 0, i, 2]

            # If the confidence is greater than 50% then we can check to see if the detected object is a person or not
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                # Checking if the detected object was a person or not
                if CLASSES[idx] != "person":
                    continue

                # If the detected object is a person then we access the coordinates of that person
                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")

                rects.append(person_box)

        # Here we will apply the non max suppression algorithm for the detection and the removal of the
        # noise in the detected persons bounding box

        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)

        # passing the bounding boxes in the non max suppression algorithm for the noise removal
        rects = non_max_suppression_fast(boundingboxes, 0.3)

        # Passing the rectangles into the tracker for tracking we will get an object containing the
        # bounding box and objectId
        objects = tracker.update(rects)

        # Iterating the objects
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # Here we will calculate the value of centroid for a particular bounding box and
            # then store that value in a dictionary

            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)

            centroid_dict[objectId] = (cX, cY, x1, y1, x2, y2)

        # This list will hold all the bounding box which are not at a safe distance from each other
        red_zone_list = []

        # Here we will be calculating the distance between all the possible combinations and if the distance
        # is greater than some threshold value then the bounding box will be green else it will be red

        for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
            # converts pixel distace (manhatan distance) to cm.
            distance = cal_dis(p1, p2)
            if distance < 180.0:
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)

                if id2 not in red_zone_list:
                    red_zone_list.append(id2)

        red_mat = []
        green_mat = []
        # Now we will display the bounding boxes accordingly
        for id, box in centroid_dict.items():
            if id in red_zone_list:
                cv2.circle(
                    frames, ((box[2] + box[4]) // 2, box[5]), 10, (0, 0, 255), -1
                )
                tmp = np.array([[[(box[2] + box[4]) // 2, box[5]]]], dtype=np.float32)
                perspective_pnt = cv2.perspectiveTransform(tmp, perspective_kernel)[0][
                    0
                ]
                red_mat.append(perspective_pnt)
                cv2.rectangle(
                    frames, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 2
                )
            else:
                cv2.circle(
                    frames, ((box[2] + box[4]) // 2, box[5]), 10, (0, 255, 0), -1
                )
                tmp = np.array([[[(box[2] + box[4]) // 2, box[5]]]], dtype=np.float32)
                perspective_pnt = cv2.perspectiveTransform(tmp, perspective_kernel)[0][
                    0
                ]
                green_mat.append(perspective_pnt)
                cv2.rectangle(
                    frames, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2
                )

        # Calculating the time difference by subtracting the start time from the end time
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = total_frames / time_diff.seconds

        fps_text = "FPS: {:.2f}".format(fps)

        cv2.putText(
            frames,
            fps_text,
            (5, 30),
            cv2.FONT_HERSHEY_TRIPLEX,
            1,
            (0, 0, 255),
            1,
        )
        cv2.imshow("Application", frames)
        # get_perspective_transform(H, W)
        image = bird_eye_view(frames, red_mat, green_mat, scale_w, scale_h)
        cv2.imshow("Bird's EYE", image)

    cap.release()
    cv2.destroyAllWindows()
