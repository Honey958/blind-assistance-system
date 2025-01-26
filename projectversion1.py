import threading
import os
import time
import cv2
import numpy as np
from model.yolo_model import YOLO

import gtts
import playsound
import os

KNOWN_DISTANCE = 40 #INCHES
PERSON_WIDTH = 16 #INCHES
MOBILE_WIDTH = 3.0 #INCHES


distanceval={"person":16.5,"chair":22,"cell phone":3.5,"tennis racket":11.4,"laptop":12,"bottle":2.7,"bowl":5,"umbrella":36.5,"suitcase":14.1,"spoon":2,"toothbrush":1,
"book":16,"bed":48,"tie":3,"scissors":4,"toothbrush":2,"hairdrier":8,"keyboard":17.17,"knife":2,"spoon":2,"fork":2,"bowl":8,"remote":5,"cup":5,"sink":15,
"diningtable":37,"sofa":72,"bench":36,"bicycle":60,"pottedplant":7,"refrigerator":22,"car":180,"bird":3,"cat":15,"tvmonitor":36,"motorbike":72,"vase":15,
"handbag":16,"backpack":12,"clock":16,"bottle":3,"fire hydrant":12,"bus":450,"boat":300,"truck":370,"toilet":20,"banana":8,"microwave":25,"traffic light":25,
"parking meter": 10,"orange":4,"teddy bear":20,"donut":6,"dog":38,"cow":73,"sports ball":15,"wine glass":5,"cake":15,"apple":5,"stop sign":19,"sandwich":6}

distancefocal= {'backpack': 2114.1612668707967, 'person': 794.7612762451172, 'bird': 3052.251663208008, 'car': 364.1760550604926, 'pottedplant': 626.6864231654575, 'cat': 877.2760963439941,
 'chair': 680.7687586004084, 'bed': 180.9578275680542, 'clock': 1042.7734879776835, 'handbag': 1349.8369791917503, 'knife': 6240.30442237854, 'laptop': 842.0555305480957, 'motorbike': 852.6407347785103,
'remote': 6996.593494415283, 'spoon': 1871.0814943313599, 'suitcase': 871.1609617192695, 'tennis racket': 2158.2359715511925, 'toothbrush': 2559.344701766968, 'tvmonitor': 1861.0940721299912, 'umbrella': 859.9984155942316,
'vase': 1153.1371815999348, 'cell phone': 819.0992684023721, 'bicycle': 1263.0791625976562, 'refrigerator': 1048.0392874912782, 'sofa': 963.9973958333334, 'diningtable': 1107.0889447186444, 'sink': 1260.6636555989583,
 'cup': 1271.5258273780346, 'bowl': 908.6286535263062, 'book': 1095.2179908752441, 'scissors': 1150.4928665161133, 'tie': 927.7920405069987, 'keyboard': 1168.441505854256,'fork': 12891.94986820221,'bottle': 2366.829888820648,
 'bench': 716.6365842819214,'fire hydrant': 1334.9302291870117,'bus': 231.5012362268236,'boat': 313.5052773157756,'truck': 284.4259922568863, 'toilet': 330.0716757774353, 'banana': 412.5487423315644, 'microwave': 717.3954662322998,
 'traffic light': 290.6974697113037, 'car': 23.06053171886338, 'parking meter': 274.96764492988586,'orange': 241.9570628553629, 'teddy bear': 330.7713496685028,'donut': 944.1684683163961,  'dog': 274.01688745147305, 'cow': 248.5910078434095,
 'sports ball': 1996.0570335388184, 'wine glass': 5308.81929397583, 'cake': 4410.844772338867, 'apple': 14141.11426448822,  'stop sign': 2263.5237844366775,'sandwich': 9884.473180770874}

distancedist= {'backpack': 55.0, 'person': 40.0, 'bird': 24.0, 'car': 70.0, 'pottedplant': 20.0, 'cat': 20.0, 'chair': 40.00000000000001,
 'bed': 40.0, 'clock': 55.0, 'handbag': 55.0, 'knife': 17.0, 'laptop': 40.0, 'motorbike': 50.0, 'remote': 20.0, 'spoon': 8.0, 'suitcase': 40.0,
  'tennis racket': 40.0, 'toothbrush': 8.0, 'tvmonitor': 70.0, 'umbrella': 40.0, 'vase': 55.0, 'cell phone': 20.0, 'bicycle': 84.0, 'refrigerator': 70.0,
   'sofa': 60.0, 'diningtable': 60.0, 'sink': 20.0, 'cup': 30.0, 'bowl': 16.0, 'book': 40.0, 'scissors': 15.7, 'tie': 20.0, 'keyboard': 30.0,'fork': 27.0,  'bottle': 40.0, 'bench': 48.0,'fire hydrant': 40.0,
'bus': 400.0, 'boat': 400.0,'truck': 400.00000000000006, 'toilet': 40.0, 'banana': 15.0, 'microwave': 40.0, 'traffic light': 120.0, 'car': 70.0,
 'parking meter': 40.0,'orange': 10.0, 'teddy bear': 40.0,'donut': 10.0, 'dog': 80.0, 'cow': 90.0,'sports ball': 40.0,'wine glass': 40.0, 'cake': 40.0, 'apple': 30.0, 'stop sign': 120.0,'sandwich': 20.0}

speech={}
respondafter= 30
runtime1 = 0
def voicemodule(functioncalltime):
    global speech
    print("voice module is called after 10 seconds",speech)
    s=""
    for i in speech:
        if len(s)==0:
            s+=" there is a {cl} in front of you at a distance of {distance:.2f} meter ".format(cl=i, distance=speech[i][1])
        else:
            s+="  and a {cl} at a distance {distance:.2f} meter ".format(cl=i, distance=speech[i][1])

    print(s)

    sound = gtts.gTTS(s,lang="en",tld="com")
    sound.save("voice/welcome.mp3")
    playsound.playsound("voice/welcome.mp3")
    os.remove(r"voice/welcome.mp3")

    speech={}

    # while(lastcalled-functioncalltime<respondafter+1):
    #     lastcalled=int(time.time())
    #     if abs(lastcalled-functioncalltime)==respondafter:
    #         print("here is data recieved by voice module")
    #         print(speech)
    #         break



def texttospeech(obj,dist,calltime):
    global runtime1
    print("txt called",runtime1,calltime,speech)

    if obj not in speech:
        speech[obj]=[1,dist]
    else:
        x=speech[obj]
        speech[obj]=[x[0]+1,dist]
    if runtime1==0:
        runtime1=calltime
    if runtime1+respondafter<calltime:
        voicemodule(int(time.time()))
        runtime1=calltime


def deleteallvoice():
    os.remove(r"voice/welcome.mp3")
    print("delete all the files from voice directory ")


def process_image(img):
    """Resize, reduce and expand image.

    # Argument:
        img: original image.

    # Returns
        image: ndarray(64, 64, 3), processed image.
    """
    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image


def get_classes(file):
    """Get classes name.

    # Argument:
        file: classes name for database.

    # Returns
        class_names: List, classes name.

    """
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names


def draw(image, boxes, scores, classes, all_classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))
        clname=all_classes[cl]
        if clname in distancefocal and clname in distancedist:
            distance= distance_finder(distancefocal[clname],distanceval[clname],w)
            distancemtr=distance*0.0254
            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f} distance = {2:.2f}m'.format(all_classes[cl], score,distancemtr),
                        (top, left - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 1,
                        cv2.LINE_AA)

            texttospeech(all_classes[cl],distancemtr,int(time.time()))




        else:
            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                        (top, left - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 1,
                        cv2.LINE_AA)

            texttospeech(all_classes[cl],0,int(time.time()))

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))

    print()


yolo = YOLO(0.6, 0.5)
file = 'data/coco_classes.txt'
all_classes = get_classes(file)

# trying
def detect_image(image, yolo, all_classes):
    pimage = process_image(image)

    start = time.time()
    boxes, classes, scores = yolo.predict(pimage, image.shape)
    end = time.time()

    print('time: {0:.2f}s'.format(end - start))
    if boxes is not None:
        draw(image, boxes, scores, classes, all_classes)
    return image


def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length

# distance finder function
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance

def frontcam( yolo, all_classes):
    camera = cv2.VideoCapture(1)
    cv2.namedWindow("Blind Assistance System", cv2.WINDOW_AUTOSIZE)
    while True:
        res, frame = camera.read()
        if not res:
            break
        image = detect_image(frame, yolo, all_classes)
        cv2.imshow("Blind Assistance System", image)

        # Save the video frame by frame
        # vout.write(image)

        if cv2.waitKey(110) & 0xff == 27:
            deleteallvoice()
            break

    # vout.release()
    camera.release()


frontcam(yolo,all_classes)
