import cv2
import numpy as np
import onnxruntime
import torch


def load_class_names(namesfile):
    with open(namesfile) as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names


def plot_boxes_cv2(img, boxes, class_names=None, color=None, line_thickness=None):
    if boxes is None:
        return img
    
    img = img.copy()
    height, width, _ = img.shape

    for box in boxes:
        box, conf, cls_id = box

        x1, y1, x2, y2 = box
        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if line_thickness:
            thickness = line_thickness
        else:
            thickness = 4

        cv2.rectangle(img, (x1, y1), (x2, y2), rgb, thickness)
        if class_names:
            cls_name = class_names[cls_id]
            ((text_w, text_h), _) = cv2.getTextSize(cls_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(img, (x1, y1 - int(1.3 * text_h)), (x1 + text_w, y1), rgb, -1)
            cv2.putText(img, cls_name, (x1, y1 - int(0.3 * text_h)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    return img


def postprocess(outputs, original_width, original_height, conf_thresh, nms_thresh):
    box_attrs = outputs[1].reshape(-1, 4)
    print('reshaped outputs:', outputs[1].reshape(-1, 4))
    boxes = []
    for box in box_attrs:
        x, y, w, h = box
        x1 = int((x - w / 2) * original_width)
        y1 = int((y - h / 2) * original_height)
        x2 = int((x + w / 2) * original_width)
        y2 = int((y + h / 2) * original_height)

        boxes.append([x1, y1, x2, y2])

    indices = cv2.dnn.NMSBoxes(boxes, [1.0] * len(boxes), conf_thresh, nms_thresh)
    return [(boxes[i], 1.0, 0) for i in indices.flatten()]


def detect(session, image_src, namesfile):
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]
    
    original_height, original_width, _ = image_src.shape

    # Input
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    #print("Shape of the network input: ", img_in.shape)

    # Compute
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: img_in})

    boxes = postprocess(outputs, original_width, original_height, 0.4, 0.6)
    return boxes




def main(onnx_file, names_file):
    session = onnxruntime.InferenceSession(onnx_file)
    class_names = load_class_names(names_file)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        boxes = detect(session, frame, names_file)
        #print(boxes)
        img_with_boxes = plot_boxes_cv2(frame, boxes, class_names=class_names)
        cv2.imshow('Detections', img_with_boxes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    onnx_file = 'Model/yolov4_-1_3_224_224_dynamic.onnx'
    names_file = 'Model/obj.names'
    main(onnx_file, names_file)
