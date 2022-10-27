
import argparse
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)



def detect(img,weights,img_size=640, 
           conf_thres=0.25, iou_thres=0.45, device="", view_img=True, 
           classes=2, agnostic_nms=True, augment=True, update=True, 
           person=True, heads=True):
    weights, view_img, weights, view_img, img_size
    # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        # ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    # save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # if webcam:
    #     view_img = check_imshow()
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    # else:
    #     save_img = True
    #     dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    # for path, img, im0s, vid_cap in dataset:
    
    
    img = letterbox(im0s, img_size=imgsz, stride=stride)[0]
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    t2 = time_synchronized()

    # Apply Classifier
    if classify:
        pred = apply_classifier(pred, modelc, img, im0s)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if webcam:  # batch_size >= 1
            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
        else:
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # img.jpg
        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            conf_arr = []
            for *xyxy, conf, cls in reversed(det):
                conf_arr.append(conf)
            # if save_txt:  # Write to file
            #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
            #     with open(txt_path + '.txt', 'a') as f:
            #         f.write(('%g ' * len(line)).rstrip() % line + '\n')
            conf = max(conf_arr)
            if save_img or view_img:  # Add bbox to image
                label = f'{names[int(cls)]} {conf:.2f}'
                if heads or person:
                    if 'head' in label and heads:
                        x1 = int(xyxy[0].item())
                        y1 = int(xyxy[1].item())
                        x2 = int(xyxy[2].item())
                        y2 = int(xyxy[3].item())
                        xmin, xmax, ymin, ymax = x1, x2, y1, y2
                        x_center = np.average([xmin, xmax])
                        y_center = np.average([ymin, ymax])
                        size = max(xmax-xmin, ymax-ymin)
                        xmin, xmax = x_center-size/2, x_center+size/2
                        ymin, ymax = y_center-size/2, y_center+size/2
                        h, w, _ = im0.shape
                        print(im0.shape)
                        if xmax > w:
                            xmin = xmin - (xmax-w)
                            xmax = w

                        if ymax > h:
                            ymin = ymin - (ymax-h)
                            ymax = h
                        cropped_img = im0[int(ymin):int(ymax),int(xmin):int(xmax)]
                        #cropped_img = im0[y1:y2, x1:x2]
                        cv2.imwrite('test.png',cropped_img)
                        #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    if 'person' in label and person:
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                else:
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

        # Print time (inference + NMS)
        print(f'{s}Done. ({t2 - t1:.3f}s)')

        # Stream results
        if view_img:
            cv2.imshow(str(p), cropped_img)
            cv2.waitKey(0)  # 1 millisecond
        
        return cropped_img
        # Save results (image with detections)
    #     if save_img:
    #         if dataset.mode == 'image':
    #             cv2.imwrite(save_path, cropped_img)
    #         else:  # 'video'
    #             if vid_path != save_path:  # new video
    #                 vid_path = save_path
    #                 if isinstance(vid_writer, cv2.VideoWriter):
    #                     vid_writer.release()  # release previous video writer

    #                 fourcc = 'mp4v'  # output video codec
    #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
    #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #                 vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    #             vid_writer.write(im0)

    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


# if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--view-img', action='store_true', help='display results')
    # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--update', action='store_true', help='update all models')
    # parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    # parser.add_argument('--name', default='exp', help='save results to project/name')
    # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # parser.add_argument('--person', action='store_true', help='displays only person')
    # parser.add_argument('--heads', action='store_true', help='displays only person')
    # opt = parser.parse_args()
    # print(opt)
    # check_requirements()

def inference(img, weights, img_size=640, conf_thres=0.65, iou_thes=0.45, device="", view_img=True, 
           agnostic_nms=True, classes=0,
           augment=True, update=False, exist_ok=True, person=True, heads=True):
    with torch.no_grad():
        if update:  # update all models (to fix SourceChangeWarning)
            for weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(img, weights, img_size, conf_thres, iou_thres, device,
                       view_img, agnostic_nms, classes, augment, update, exist_ok, person, heads)
                strip_optimizer(weights)
        else:
            detect(img, weights, img_size, conf_thres, iou_thres, device,
                       view_img, agnostic_nms, classes, augment, update, exist_ok, person, heads)
