import face_recognition
import cv2
from datetime import datetime, timedelta
import numpy as np
import platform
import pickle
from flask import Flask, render_template, Response
import nanocamera as nano
import dlib
print(dlib.DLIB_USE_CUDA)

# Flask initialize
app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

# Our list of known face encodings and a matching list of metadata about each face.
known_face_encodings = []
known_face_metadata = []


def save_known_faces():
    with open("known_faces.dat", "wb") as face_data_file:
        face_data = [known_face_encodings, known_face_metadata]
        pickle.dump(face_data, face_data_file)
        print("Known faces backed up to disk.")


def load_known_faces():
    global known_face_encodings, known_face_metadata

    try:
        with open("known_faces.dat", "rb") as face_data_file:
            known_face_encodings, known_face_metadata = pickle.load(face_data_file)
            print("Known faces loaded from disk.")
    except FileNotFoundError as e:
        print("No previous face data found - starting with a blank known face list.")
        pass


def running_on_jetson_nano():
    # To make the same code work on a laptop or on a Jetson Nano, we'll detect when we are running on the Nano
    # so that we can access the camera correctly in that case.
    # On a normal Intel laptop, platform.machine() will be "x86_64" instead of "aarch64"
    return platform.machine() == "aarch64"


def open_cam_rtsp(uri, width, height, latency):
    gst_str = ('rtspsrc location={} latency={} ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(uri, latency, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def open_cam_usb(dev, width, height):
    # We want to set width and height here, otherwise we could just do:
    #     return cv2.VideoCapture(dev)
    #gst_str = ('v4l2src device=/dev/video{} ! '
    #           'video/x-raw, width=(int){}, height=(int){} ! '
    #           'videoconvert ! appsink').format(dev, width, height)
    #gst_str = ('v4l2src device=/dev/video{} ! '
    #           'image/jpeg, width=(int){}, height=(int){}, framerate=30/1 ! jpegparse ! '
    #           'nvjpegdec ! video/x-raw, format=(string)I420 ! nvvidconv ! '
    #           'video/x-raw, format=(string)RGBA ! nvvidconv ! video/x-raw, format=(string)RGBA ! appsink').format(dev, width, height)
    gst_str = ('v4l2src device=/dev/video{} ! video/x-raw, width=(int){}, height=(int){} ! ' 
               'nvvidconv ! video/x-raw(memory:NVMM), format=(string)I420, width=(int){}, height=(int){} ! '
               'nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! '
               'videoconvert ! tee ! appsink').format(dev, width, height, width, height, width, height)
#'video/x-raw(memory:NVMM), width=(int){}, height=(int){}, format=UYVY ! '
#               'nvvidconv ! video/x-raw, width=(int){}, height=(int){}, framerate=(fraction)60/1 ! videoconvert ! appsink').format(dev, width, height, width, height)

    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def open_cam_onboard(width, height):
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'nvcamerasrc' in gst_elements:
        # On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
        gst_str = ('nvcamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)2592, height=(int)1458, '
                   'format=(string)I420, framerate=(fraction)30/1 ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    elif 'nvarguscamerasrc' in gst_elements:
        gst_str = ('nvarguscamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)1920, height=(int)1080, '
                   'format=(string)NV12, framerate=(fraction)30/1 ! '
                   'nvvidconv flip-method=2 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    else:
        raise RuntimeError('onboard camera source not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def get_jetson_gstreamer_source(RTSP_ADDR, capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=25, flip_method=0):
    """
    Return an OpenCV-compatible video source description that uses gstreamer to capture video from the camera on a Jetson Nano
    """
    return (
            #'nvarguscamerasrc ! video/x-raw(memory:NVMM), ' +
            #'width=(int){}, height=(int){}, '.format(capture_width, capture_height) +
            #'format=(string)NV12, framerate=(fraction){}/1 ! '.format(framerate) +
            #'nvvidconv flip-method={} ! '.format(flip_method) +
            #'video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! '.format(display_width, display_height) +
            #'videoconvert ! video/x-raw, format=(string)BGR ! appsink'
           
            #"rtspsrc location=rtsp://184.72.239.149/vod/mp4:BigBuckBunny_115k.mov ! application/x-rtp, media=video ! ";
            #"decodebin ! 
            #"video/x-raw, format=NV12 ! videoconvert ! 
            # video/x-raw, width=(int)" << mWidth << ", height=(int)" << mHeight << ", format=RGB ! videoconvert ! 
            # video/x-raw, format=RGB ! videoconvert !";
            #"appsink name=mysink";
    
            ##---- RTSP works confirmed1 ----
            #'rtspsrc location={} latency=500 ! '.format(RTSP_ADDR) +
            #'rtph264depay ! h264parse ! ' + 
            #'omxh264dec disable-dvfs=1 ! videoconvert ! ' +
            #'appsink'
            ##------------------------------
            
            ###---- RTSP works with nvv4l2decoder ----
            ##'rtspsrc location={} latency=500 ! '.format(RTSP_ADDR) +
            ##'rtph264depay ! h264parse ! ' + 
            ##'nvv4l2decoder ! nvvidconv ! ' + 
            ##'appsink'
            ##'video/x-raw(memory:NVMM),format=BGRx,width={},height={} ! '.format(display_width,display_height) +
            ##'queue ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! videorate max-rate={} drop-only=true average-period=5000000 ! '.format(framerate) +
            ##'video/x-raw,framerate={}/1 ! appsink sync=true'.format(framerate)
            )


def register_new_face(face_encoding, face_image):
    """
    Add a new person to our list of known faces
    """
    # Add the face encoding to the list of known faces
    known_face_encodings.append(face_encoding)
    # Add a matching dictionary entry to our metadata list.
    # We can use this to keep track of how many times a person has visited, when we last saw them, etc.
    known_face_metadata.append({
        "first_seen": datetime.now(),
        "first_seen_this_interaction": datetime.now(),
        "last_seen": datetime.now(),
        "seen_count": 1,
        "seen_frames": 1,
        "face_image": face_image,
    })


def lookup_known_face(face_encoding):
    """
    See if this is a face we already have in our face list
    """
    metadata = None

    # If our known face list is empty, just return nothing since we can't possibly have seen this face.
    if len(known_face_encodings) == 0:
        return metadata

    # Calculate the face distance between the unknown face and every face on in our known face list
    # This will return a floating point number between 0.0 and 1.0 for each known face. The smaller the number,
    # the more similar that face was to the unknown face.
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

    # Get the known face that had the lowest distance (i.e. most similar) from the unknown face.
    best_match_index = np.argmin(face_distances)

    # If the face with the lowest distance had a distance under 0.6, we consider it a face match.
    # 0.6 comes from how the face recognition model was trained. It was trained to make sure pictures
    # of the same person always were less than 0.6 away from each other.
    # Here, we are loosening the threshold a little bit to 0.65 because it is unlikely that two very similar
    # people will come up to the door at the same time.
    if face_distances[best_match_index] < 0.65:
        # If we have a match, look up the metadata we've saved for it (like the first time we saw it, etc)
        metadata = known_face_metadata[best_match_index]

        # Update the metadata for the face so we can keep track of how recently we have seen this face.
        metadata["last_seen"] = datetime.now()
        metadata["seen_frames"] += 1

        # We'll also keep a total "seen count" that tracks how many times this person has come to the door.
        # But we can say that if we have seen this person within the last 5 minutes, it is still the same
        # visit, not a new visit. But if they go away for awhile and come back, that is a new visit.
        if datetime.now() - metadata["first_seen_this_interaction"] > timedelta(minutes=5):
            metadata["first_seen_this_interaction"] = datetime.now()
            metadata["seen_count"] += 1

    return metadata


def threadVid():
    # Get access to the webcam. The method is different depending on if this is running on a laptop or a Jetson Nano.
    #if running_on_jetson_nano():
    #    #RTSP_ADDR = 'rtsp://admin:intflow3121@192.168.0.100:554/cam/realmonitor?channel=1&subtype=0'
    #    # Accessing the camera with OpenCV on a Jetson Nano requires gstreamer with a custom gstreamer source string
    #    #print(get_jetson_gstreamer_source(RTSP_ADDR))
    #    #video_capture = cv2.VideoCapture(get_jetson_gstreamer_source(0), cv2.CAP_GSTREAMER)
    #    video_capture = open_cam_usb(1, 320, 240)
    #else:
    #    # Accessing the camera with OpenCV on a laptop just requires passing in the number of the webcam (usually 0)
    #    # Note: You can pass in a filename instead if you want to process a video file instead of a live camera stream
    #    video_capture = cv2.VideoCapture(0)

    #video_capture = cv2.VideoCapture(RTSP_ADDR)
    #video_capture = cv2.VideoCapture(0)
    #video_capture = cv2.VideoCapture("rtsp://admin:intflow3121@192.168.0.100:554/cam/realmonitor?channel=1&subtype=0")
    #video_capture_thermal = cv2.VideoCapture("rtsp://admin:intflow3121@192.168.0.100:554/cam/realmonitor?channel=2&subtype=0")
    video_capture = nano.Camera(camera_type=1, device_id=0, width=640, height=480, fps=30, enforce_fps=True)

    # Track how long since we last saved a copy of our known faces to disk as a backup.
    number_of_faces_since_save = 0

    while video_capture.isReady():
        # Grab a single frame of video
        frame = video_capture.read()
        #ret_thermal, frame_thermal = video_capture_thermal.read()

        #video_capture.grab()
        #video_capture_thermal.grab()

        #if ret == 0:# or ret_thermal == 0:
        #    continue

        #frame = cv2.resize(frame, (640, 480))
        #frame_thermal = cv2.resize(frame_thermal, (480, 640))

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the face locations and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame, model='cnn')
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Loop through each detected face and see if it is one we have seen before
        # If so, we'll give it a label that we'll draw on top of the video.
        face_labels = []
        for face_location, face_encoding in zip(face_locations, face_encodings):
            # See if this face is in our list of known faces.
            metadata = lookup_known_face(face_encoding)

            # If we found the face, label the face with some useful information.
            if metadata is not None:
                time_at_door = datetime.now() - metadata['first_seen_this_interaction']
                face_label = "At door {}s".format(int(time_at_door.total_seconds()))

            # If this is a brand new face, add it to our list of known faces
            else:
                face_label = "New visitor!"

                # Grab the image of the the face from the current frame of video
                top, right, bottom, left = face_location
                face_image = small_frame[top:bottom, left:right]
                face_image = cv2.resize(face_image, (150, 150))

                # Add the new face to our known face data
                register_new_face(face_encoding, face_image)

            face_labels.append(face_label)

        # Draw a box around each face and label each face
        for (top, right, bottom, left), face_label in zip(face_locations, face_labels):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, face_label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # Display recent visitor images
        number_of_recent_visitors = 0
        for metadata in known_face_metadata:
            # If we have seen this person in the last minute, draw their image
            if datetime.now() - metadata["last_seen"] < timedelta(seconds=10) and metadata["seen_frames"] > 5:
                # Draw the known face image
                x_position = number_of_recent_visitors * 150
                frame[30:180, x_position:x_position + 150] = metadata["face_image"]
                number_of_recent_visitors += 1

                # Label the image with how many times they have visited
                visits = metadata['seen_count']
                visit_label = "{} visits".format(visits)
                if visits == 1:
                    visit_label = "First visit"
                cv2.putText(frame, visit_label, (x_position + 10, 170), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        if number_of_recent_visitors > 0:
            cv2.putText(frame, "Visitors at Door", (5, 18), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)


        # Yield video frame to flask
        #frame_concat = np.concatenate((frame,frame_thermal),axis=1)
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
				   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

        # Display the final frame of video with boxes drawn around each detected fames
        ##cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    save_known_faces()
        #    break

        # We need to save our known faces back to disk every so often in case something crashes.
        if len(face_locations) > 0 and number_of_faces_since_save > 100:
            save_known_faces()
            number_of_faces_since_save = 0
        else:
            number_of_faces_since_save += 1

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


#Flask thread run
@app.route('/video_feed')
def video_feed():
	return Response(threadVid(),
					mimetype='multipart/x-mixed-replace; boundary=frame')			

def main():
    app.run(host='0.0.0.0', debug=False)


if __name__ == "__main__":
    load_known_faces()
    main()
