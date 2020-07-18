import cv2
import numpy as np
import glob

WINDOW_NAME = "YOLO train"
TARGET_DIR = "../train"

mousePt = []
rectPt = []
imgpath = None
img = None

def getParam():
    global retPt

    if len(mousePt) < 2 or len(mousePt[0]) < 2:
        return 0, 0, 0, 0

    w = int(abs(mousePt[0][0]-mousePt[1][0]))
    h = int(abs(mousePt[0][1]-mousePt[1][1]))
    cx = int(mousePt[0][0] + w/2)
    if mousePt[0][0] > mousePt[1][0]:
        cx = int(mousePt[1][0] + w/2)
    cy = int(mousePt[0][1] + h/2)
    if mousePt[0][1] > mousePt[1][1]:
        cy = int(mousePt[1][1] + h/2)

    return cx, cy, w, h


def showRectangle(name, wait=False):
    global rectPt, imgpath, img
    height, width = img.shape[:2]
    print("{}: {}x{}, rect size: {}".format(imgpath, width, height, len(rectPt)))
    for rect in rectPt:
        cx, cy, w, h = rect

        print("{} {} {} {} {}".format(0, cx/width, cy/height, w/width, h/height))
        cv2.rectangle(img,
                    (int(cx-w/2), int(cy-h/2)),
                    (int(cx+w/2), int(cy+h/2)),
                    (255, 139, 0),
                    int(width/200))

    cv2.imshow(name, img)
    cv2.setMouseCallback(name, mouse_event)
    ret = cv2.waitKeyEx(100)
    while wait and ret == -1:
        ret = cv2.waitKeyEx(100)
    return ret

def mouse_event(event, x, y, flags, param):
    global mousePt, rectPt, imgpath, img
    if event == cv2.EVENT_LBUTTONDOWN:
        mousePt = [(x,y)]
    elif event == cv2.EVENT_LBUTTONUP:
        mousePt.append((x,y))
        rectPt.append(getParam())
        showRectangle(WINDOW_NAME)
    elif event == cv2.EVENT_RBUTTONDBLCLK:
        height,width = img.shape[:2]
        save_path = imgpath[:imgpath.rfind('.')+1] + "txt"
        with open(save_path, 'a') as out:
            for rect in rectPt:
                cx, cy, w, h = rect
                outstr = "0 {} {} {} {}\n".format(cx/width, cy/height, w/width, h/height)
                out.write(outstr)
                print("save to {}:{}".format(save_path, outstr))

if __name__ == '__main__':
    images = sorted(glob.glob("{}/*.jpg".format(TARGET_DIR)))
    print("image len = {}. start".format(len(images)))
    i = 0
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    while True:
        if i >= len(images):
            break
        imgpath = images[i]
        img = cv2.imread(imgpath)
        key = showRectangle(WINDOW_NAME, True)
        if key == 27: # esc to exit
            break
        elif key == 2424832: # left arrow to back
            i -= 1
            rectPt = []
        elif key == 2555904 : # right arrow to next
            i += 1
            rectPt = []
        elif key == 8: # backspace to remove head of mousePt
            remId = len(rectPt)-1
            if remId >= 0:
                rectPt.pop(len(rectPt)-1)

    cv2.destroyAllWindows()
    print("finish")
