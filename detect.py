import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import os
from google.colab.patches import cv2_imshow
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(source,weights,imgsz=640, 
           conf_thres=0.25, iou_thres=0.45, device="", view_img=False, save_img=True,
           classes=2,project='runs/detect',name = 'exp', agnostic_nms=True, augment=True, update=True, 
           exist_ok =False, person=True, heads=True, save_txt =False):
    source, weights, view_img, save_txt, imgsz
    # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    #     ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    current_directory = os.getcwd()
    save_dir = os.path.join(current_directory, r'results/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # save_dir = Path(increment_path(Path(project) / name, exist_ok))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    cropped_img_arr = []
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
    save_img = True
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    count = 0
    for path, img, im0s in dataset:
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
            # if webcam:  # batch_size >= 1
            #     p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            # else:
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            #p = Path(p)  # to Path
            save_path = str(save_dir+str(count)+'.png')  # img.jpg
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                save_conf = True
                save_img = True
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        if heads or person:
                            if 'head' in label and heads:
                                x1 = int(xyxy[0].item())
                                y1 = int(xyxy[1].item())
                                x2 = int(xyxy[2].item())
                                y2 = int(xyxy[3].item())
                                xmin, xmax, ymin, ymax = x1, x2, y1, y2
                                xmin -= abs(int(0.15 * (xmax - xmin)))
                                xmax += abs(int(0.15 * (xmax - xmin)))
                                ymin -= abs(int(0.15 * (ymax - ymin)))
                                ymax += abs(int(0.15 * (ymax - ymin)))
                                xmin, xmax, ymin, ymax = abs(xmin), abs(xmax), abs(ymin), abs(ymax)
                                
                                x_center = np.average([xmin, xmax])
                                y_center = np.average([ymin, ymax])
                                size = max((xmax-xmin), (ymax-ymin))
                               
                                xmin, xmax = x_center-size/2, x_center+size/2
                                ymin, ymax = y_center-size/2, y_center+size/2
                               
                                h, w, _ = im0.shape
                                if xmax >= w:
                                    xmin = xmin - (xmax-w)
                                    xmax = w

                                if ymax >= h:
                                    ymin = ymin - (ymax-h)
                                    ymax = h
                               
                                print(int(xmin), int(xmax), int(ymin), int(ymax))
                                cropped_img = im0[int(ymin):int(ymax),int(xmin):int(xmax)]
                                #cropped_img = im0[y1:y2, x1:x2]
                                #cv2.imwrite('test3.png',cropped_img)
                                #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                            if 'person' in label and person:
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        else:
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
            count+=1
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            cropped_img_arr.append(cropped_img)
            # Stream results
            if view_img:
                cv2_imshow(cropped_img)
                cv2.waitKey(0)  # 1 millisecond
            
            
            #Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, cropped_img)
             # if save_txt or save_img:
             #    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
             #    print(f"Results saved to {save_dir}{s}")
    
    print(f'Done. ({time.time() - t0:.3f}s)')
    #print(cropped_img_arr)
    return cropped_img_arr



def inference(source, weights, img_size=640, conf_thres=0.65, iou_thres=0.45, device="", view_img=False, save_img=True, 
           agnostic_nms=True, classes=0, project='runs/detect', name = 'exp',
           augment=True, update=False, exist_ok=False, person=True, heads=True, save_txt =False):
    with torch.no_grad():
        if update:  # update all models (to fix SourceChangeWarning)
            for weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                result = detect(source, weights, img_size, conf_thres, iou_thres, device,
                       view_img,save_img, agnostic_nms, classes, project, name, augment, update, exist_ok, person, heads, save_txt)
                strip_optimizer(weights)
        else:
            result = detect(source, weights, img_size, conf_thres, iou_thres, device,
                       view_img, save_img, agnostic_nms, classes,project, name, augment, update,exist_ok, person, heads,save_txt)
    return result