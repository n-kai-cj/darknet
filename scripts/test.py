import cv2
import numpy as np
import glob
import time

inW = 416
inH = 416
classesFile = "../class.names"
conf = "../yolov4-custom.cfg"
model = "../backup/yolov4-custom_last.weights"
confThreshold = 0.5
target_dir="C:/darknet/img"
cuda = True

def postprocess(frame, outs, outLayerType, classes):

    height,width = frame.shape[:2]
    boxes = []
    confidences = []
    classIds = []
    if outLayerType == 'Region':
        for out in outs:
            for i, data in enumerate(out):
                scores = data[5:]
                classId = np.argmax(scores)
                confidence = float(scores[classId])
                if confThreshold < confidence:
                    box = data[0:4] * np.array([width, height, width, height])
                    cx = int(box[0])
                    cy = int(box[1])
                    w = int(box[2])
                    h = int(box[3])
                    left = int(cx - w / 2)
                    top = int(cy - h / 2)
                    boxes.append([left, top, w, h])
                    confidences.append(confidence)
                    classIds.append(classId)

    # apply non-maximum suppression to suppress weak, overlapping bboxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, 0.4)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (left, top, w, h) = (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])
            confidence = confidences[i]
            classId = classIds[i]
            # draw bbox rectangle and label on the image
            cv2.rectangle(frame,
                          (left, top),
                          (left+w, top+h),
                          (255,128*float(i),0),2)
            label = "{} {:.2f}".format(classes[classId], confidence)
            labelSize, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            cv2.rectangle(frame,
                          (left, top-labelSize[1]),
                          (left+labelSize[0], top+baseline),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label,
                        (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))



if __name__ == '__main__':
    print("--- start ---")

    classes = None
    with open(classesFile, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    net = cv2.dnn.readNetFromDarknet(conf, model)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA if cuda else cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA if cuda else cv2.dnn.DNN_TARGET_CPU)
    outNames = net.getUnconnectedOutLayersNames()
    outLayers = net.getLayerId(net.getLayerNames()[-1])
    outLayerType = net.getLayer(outLayers).type
    print("read net succeed")

    images = sorted(glob.glob(e) for e in ['{}/*.jpg'.format(target_dir), '{}/*.png'.format(target_dir), '{}/*.bmp'.format(target_dir)])
    images = np.array([e for e in images if e != []]).flatten()

    seqFlag = False
    imgId = 0
    maxImgId = len(images)

    while True:
        if imgId >= maxImgId:
            break
        image_path = images[imgId]
        frame = cv2.imread(image_path)

        start_time = time.time()

        # Create a 4D blob from a frame
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (inW, inH),
                                     swapRB=True, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(outNames)

        inf_time = time.time()

        tick_freq = cv2.getTickFrequency()
        t, layersTimings = net.getPerfProfile()
        time_ms = t / tick_freq * 1000.0
        #print("inference time: {:.1f}[ms]".format(time_ms))

        # Showing information on the screen
        postprocess(frame, outs, outLayerType, classes)

        post_time = time.time()
        print("[{:05d}/{:05d}] infer={:.3f}[sec], postproc={:.3f}[sec]".format(imgId+1, maxImgId, inf_time-start_time, post_time-inf_time))

        cv2.imshow("YOLO", frame)
        key = cv2.waitKeyEx(10)
        while key == -1 and not seqFlag:
            key = cv2.waitKeyEx(1000)
        if key == 27: # esc to exit
            break
        elif key == 32: # space to change continuous mode
            seqFlag = not seqFlag
        elif key == 2424832: # left arrow to back
            imgId -= 2
        
        # increment image id
        imgId += 1

    cv2.destroyAllWindows()

    print("--- finish ---")
