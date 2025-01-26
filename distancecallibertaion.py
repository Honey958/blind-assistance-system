import os
import time
import cv2
import numpy as np
from model.yolo_model import YOLO
KNOWN_DISTANCE = 40 #INCHES
PERSON_WIDTH = 16 #INCHES
MOBILE_WIDTH = 3.0 #INCHES



distancefocal={}
distancedist={}
distanceval={
"person":16.5,"chair":22,"cell phone":3.5,"tennis racket":11.4,"laptop":12,"bottle":2.7,"bowl":5,"umbrella":36.5,"suitcase":14.1,"spoon":1,"toothbrush":1,
"book":16,"bed":48,"tie":3,"scissors":4,"toothbrush":2,"hairdrier":8,"keyboard":17.17,"knife":2,"spoon":2,"fork":2,"bowl":8,"remote":5,"cup":5,"sink":15,
"dining table":37,"sofa":72,"bench":36,"bicycle":60,"pottedplant":7,"refrigerator":22,"car":180,"bird":3,"cat":15,"tvmonitor":36,"motorbike":72,"vase":15,
"handbag":16,"bagpack":12,"clock":16,"bottle":3}




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
        datalist=[]
        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))
        datalist.append(box)
        datalist.append(cl)
        datalist.append((x,y-2))
        print(datalist)
        distancemapper(datalist)
    print()


yolo = YOLO(0.6, 0.5)
file = 'data/coco_classes.txt'
all_classes = get_classes(file)

# trying
def detect_image_dist(image, yolo, all_classes):
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




# focal_person=focal_length_finder(KNOWN_DISTANCE,PERSON_WIDTH,)

def distancemapper(datalist):
    print(datalist)
    print("width",datalist[0][2])
    print('classes=',datalist[1])
    width= datalist[0][2]
    cl=datalist[1]
    clname=all_classes[datalist[1]]
    x,y = datalist[2]
    print(cl)
    print(all_classes[cl])
    if all_classes[cl]=="person":
        widthperson=distanceval['person']
        focal=focal_length_finder(KNOWN_DISTANCE,widthperson,width)
        dist=distance_finder(focal,widthperson,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("distance=",dist)
        print("focal=",focal)

    if all_classes[cl]=="chair":
        widthchair=distanceval['chair']
        focal=focal_length_finder(KNOWN_DISTANCE,widthchair,width)
        dist=distance_finder(focal,widthchair,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("focal=",focal)

        print("distance=",dist)
    if all_classes[cl]=="cell phone":
        widthphone=distanceval['cell phone']
        focal = focal_length_finder(20,widthphone,width)
        dist= distance_finder(focal,widthphone,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("focal=",focal)

        print("distance=",dist)
    if all_classes[cl]=="tennis racket":
        widthtennis=distanceval['tennis racket']
        focal=focal_length_finder(KNOWN_DISTANCE,widthtennis,width)
        dist= distance_finder(focal,widthtennis,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("focal=",focal)

        print("distance=",dist)
    if all_classes[cl]=="laptop":
        widthlaptop=distanceval['laptop']
        focal = focal_length_finder(KNOWN_DISTANCE,widthlaptop,width)
        dist= distance_finder(focal,widthlaptop,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("focal=",focal)

        print("distance=",dist)
    if all_classes[cl]=="bottle":
        widthbottle=distanceval['bottle']
        focal=focal_length_finder(KNOWN_DISTANCE,widthbottle,width)
        dist= distance_finder(focal,widthbottle,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("focal=",focal)

        print("distance=",dist)
    if all_classes[cl]=="bed":
        widthbed=distanceval['bed']
        focal=focal_length_finder(KNOWN_DISTANCE,widthbed,width)
        dist= distance_finder(focal,widthbed,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("focal=",focal)

        print("distance=",dist)


        print("distance=",dist)
    if all_classes[cl]=="umbrella":
        widthumbrella=distanceval['umbrella']
        focal=focal_length_finder(KNOWN_DISTANCE,widthumbrella,width)
        dist= distance_finder(focal, widthumbrella,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("focal=",focal)

        print("distance=",dist)
    if all_classes[cl]=="suitcase":
        widthsuitcase=distanceval['suitcase']
        focal=focal_length_finder(KNOWN_DISTANCE,widthsuitcase,width)
        dist = distance_finder(focal, widthsuitcase,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("focal=",focal)

        print("distance=",dist)

    if all_classes[cl]=="toothbrush":
        widthbrush=distanceval['toothbrush']
        focal=focal_length_finder(KNOWN_DISTANCE,widthbrush,width)
        dist = distance_finder(focal, widthbrush,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("focal=",focal)

        print("distance=",dist)
    if all_classes[cl]=="spoon":
        widthspoon=distanceval['spoon']
        focal=focal_length_finder(KNOWN_DISTANCE,widthspoon,width)
        dist = distance_finder(focal, widthspoon,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("focal=",focal)
        print("distance=",dist)
    if all_classes[cl]=="scissors":
        widthscissors=distanceval['scissors']
        focal=focal_length_finder(15.748,widthscissors,width)
        dist = distance_finder(focal, widthscissors,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("focal=",focal)

        print("distance=",dist)
    if all_classes[cl]=="book":
        widthbook=distanceval['book']
        focal=focal_length_finder(KNOWN_DISTANCE,widthbook,width)
        dist = distance_finder(focal, widthbook,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("focal=",focal)

        print("distance=",dist)

    if all_classes[cl]=="hair drier":
        widthdrier=distanceval['hair drier']
        focal=focal_length_finder(20,widthdrier,width)
        dist= distance_finder(focal,widthdrier,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("focal=",focal)

        print("distance=",dist)

    if all_classes[cl]=="keyboard":
        widthkeyboard=distanceval['keyboard']
        focal=focal_length_finder(30,widthkeyboard,width)
        dist= distance_finder(focal,widthkeyboard,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("focal=",focal)

        print("distance=",dist)

    if all_classes[cl]=="knife":
        widthact=distanceval['knife']
        focal=focal_length_finder(17,widthact,width)
        dist= distance_finder(focal,widthact,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("focal=",focal)

        print("distance=",dist)
    if all_classes[cl]=="spoon":
        widthact=distanceval['spoon']
        focal=focal_length_finder(27,widthact,width)
        dist= distance_finder(focal,widthact,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("focal=",focal)

        print("distance=",dist)
    if all_classes[cl]=="fork":
        widthact=distanceval['fork']
        focal=focal_length_finder(27,widthact,width)
        dist= distance_finder(focal,widthact,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("focal=",focal)

        print("distance=",dist)
    if all_classes[cl]=="bowl":
        widthact=distanceval['bowl']
        focal=focal_length_finder(16,widthact,width)
        dist= distance_finder(focal,widthact,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("focal=",focal)
        print("distance=",dist)
    if all_classes[cl]=="cup":
        widthact=distanceval['cup']
        focal=focal_length_finder(30,widthact,width)
        dist= distance_finder(focal,widthact,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("focal=",focal)
        print("distance=",dist)

    if all_classes[cl]=="sink":
        widthact=distanceval['sink']
        focal=focal_length_finder(20,widthact,width)
        dist= distance_finder(focal,widthact,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("focal=",focal)
        print("distance=",dist)

    if all_classes[cl]=="dining table":
        widthact=distanceval['dining table']
        focal=focal_length_finder(60,widthact,width)
        dist= distance_finder(focal,widthact,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("focal=",focal)
        print("distance=",dist)

    if all_classes[cl]=="sofa":
        widthact=distanceval['sofa']
        focal=focal_length_finder(60,widthact,width)
        dist= distance_finder(focal,widthact,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("focal=",focal)
        print("distance=",dist)

    if all_classes[cl]=="bench":
        widthact=distanceval['bench']
        focal=focal_length_finder(48,widthact,width)
        dist= distance_finder(focal,widthact,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("focal=",focal)
        print("distance=",dist)

    if all_classes[cl]=="bicycle":
        widthact=distanceval['bicycle']
        focal=focal_length_finder(84,widthact,width)
        dist= distance_finder(focal,widthact,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("focal=",focal)
        print("distance=",dist)

    if all_classes[cl]=="potted plant ":
        widthact=distanceval['potted plant']
        focal=focal_length_finder(28,widthact,width)
        dist= distance_finder(focal,widthact,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("focal=",focal)
        print("distance=",dist)

    if all_classes[cl]=="refrigerator":
        widthact=distanceval['refrigerator']
        focal=focal_length_finder(70,widthact,width)
        dist= distance_finder(focal,widthact,width)
        if clname not in distancedist and clname not in distancefocal:
            distancedist[clname]=dist
            distancefocal[clname]=focal
        print("focal=",focal)
        print("distance=",dist)










for (root, dirs, files) in os.walk('images/test'):
    if files:
        for f in files:
            print(f)
            path = os.path.join(root, f)
            image = cv2.imread(path)
            image= detect_image_dist(image, yolo, all_classes)
            # distancemapper(data)
            cv2.imwrite('images/res/' + f, image)




print("distanceval",distanceval)
print("distancefocal",distancefocal)
print("distancedist",distancedist)











# trying end
# frontcam(yolo,all_classes)
