from itertools import count
from reid.config import cfg as reidCfg
from reid.modeling import build_model
from reid.data.transforms import build_transforms
from reid.data import make_data_loader
from utils.utilsReid import *
from utils.torch_utils import select_device, time_sync
from utils.plots import Annotator, colors, save_one_box, save_one_img
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from models.common import DetectMultiBackend
import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import goto


def init_person_reid():
    FILE = Path(__file__).resolve()
    # print(FILE)
    ROOT = FILE.parents[0]  # YOLOv5 root directory
    # print(ROOT)

    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
    # print(ROOT)

    source = './data/video/test.mp4'
    nosave = False
    save_img = not nosave and not source.endswith(
        '.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)

    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    project = ROOT / 'results'  # save results to project/name
    name = 'exp'  # save results to project/name
    exist_ok = False  # existing project/name ok, do not increment
    save_txt = False  # save results to *.txt
    save_dir = increment_path(Path(project) / name,
                              exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                          exist_ok=True)  # make dir

    # Load yolo model
    device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    device = select_device(device)
    weights = ROOT / 'weights/newbestcacbam.pt'  # model.pt path(s)
    dnn = False  # use OpenCV DNN for ONNX inference
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    # imgsz=[1280] # inference size (pixels)
    imgsz = [1280, 1280]  # inference size (pixels)
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    ############# 行人重识别模型初始化 #############
    query_loader, num_query = make_data_loader(reidCfg)
    reidModel = build_model(reidCfg, num_classes=10126)
    reidModel.load_param(reidCfg.TEST.WEIGHT)
    reidModel.to(device).eval()

    query_feats = []
    query_pids = []

    for i, batch in enumerate(query_loader):
        with torch.no_grad():
            img, pid, camid = batch
            img = img.to(device)
            # 一共2张待查询图片，每张图片特征向量2048 torch.Size([2, 2048])
            feat = reidModel(img)
            query_feats.append(feat)
            # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            query_pids.extend(np.asarray(pid))

    query_feats = torch.cat(query_feats, dim=0)  # torch.Size([2, 2048])
    print("The query feature is normalized")
    query_feats = torch.nn.functional.normalize(
        query_feats, dim=1, p=2)  # 计算出查询图片的特征向量

    # Half
    half = False  # use FP16 half-precision inference
    # half precision only supported by PyTorch on CUDA
    half &= pt and device.type != 'cpu'
    if pt:
        model.model.half() if half else model.model.float()

    # Dataloader

    save_img = True

    dataset = LoadImages(source, img_size=imgsz,
                         stride=stride, auto=pt and not jit)
    bs = 1  # batch_size

    vid_path, vid_writer = [None] * bs, [None] * bs

    colorss = [[random.randint(0, 255) for _ in range(3)]
               for _ in range(10)]  # 对于每种类别随机使用一种颜色画框

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(
            1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    return device, half, save_dir, model, dataset, names, save_img, \
        reidModel, query_feats, colorss, vid_path, vid_writer


def process_a_img(index, count, device, half, save_dir, model, dataset, names, save_img,
                  reidModel, query_feats, colorss, vid_path, vid_writer, danger_img, danger_first, danger_second, danger_nums):
    # path, im, im0s, vid_cap, s = dataset[index]
    path, im, im0s, vid_cap, s = dataset[index]
    visualize = False  # visualize features
    dt, seen = [0.0, 0.0, 0.0], 0

    conf_thres = 0.4  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    dist_thres = 1.5
    max_det = 10  # maximum detections per image
    view_img = True  # show results
    save_txt = True  # save results to *.txt
    save_conf = True  # save confidences in --save-txt labels
    save_crop = True  # save cropped prediction boxes
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False  # class-agnostic NMS
    augment = False  # augmented inference

    line_thickness = 0  # bounding box thickness (pixels)
    hide_labels = False  # hide labels
    hide_conf = False  # hide confidences

    t1 = time_sync()
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1

    # Inference
    visualize = increment_path(
        save_dir / Path(path).stem, mkdir=True) if visualize else False
    pred = model(im, augment=augment, visualize=visualize)
    t3 = time_sync()
    dt[1] += t3 - t2

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes)
    dt[2] += time_sync() - t3

    # Second-stage classifier (optional)
    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

    # Process predictions

    for i, det in enumerate(pred):  # per image

        seen += 1

        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # im.jpg
        txt_path = str(save_dir / 'labels' / p.stem) + \
            ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
        s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if save_crop else im0  # for save_crop
        annotator = Annotator(
            im0, line_width=line_thickness, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Draw bounding boxes and labels of detections
            # (x1y1x2y2, obj_conf, class_conf, class_pred)

            gallery_img = []
            gallery_loc = []

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # if save_txt:  # Write to file
                #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                #     with open(txt_path + '.txt', 'a') as f:
                #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (
                        names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    # annotator.box_label(xyxy, label, color=colors(c, True))
                    # if save_crop:
                    #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # if classes[int(cls)] == 'person':
                if int(cls) == 0:
                    # plot_one_bo x(xyxy, im0, label=label, color=colors[int(cls)])
                    xmin = int(xyxy[0])
                    ymin = int(xyxy[1])
                    xmax = int(xyxy[2])
                    ymax = int(xyxy[3])
                    w = xmax - xmin  # 233
                    h = ymax - ymin  # 602
                    # 如果检测到的行人太小了，感觉意义也不大
                    # 这里需要根据实际情况稍微设置下
                    if w * h > 500:
                        gallery_loc.append((xmin, ymin, xmax, ymax))
                        # HWC (602, 233, 3)
                        crop_img = im0[ymin:ymax, xmin:xmax]
                        crop_img = Image.fromarray(cv2.cvtColor(
                            crop_img, cv2.COLOR_BGR2RGB))  # PIL: (233, 602)
                        crop_img = build_transforms(reidCfg)(crop_img).unsqueeze(
                            0)  # torch.Size([1, 3, 256, 128])
                        gallery_img.append(crop_img)

            if gallery_img:
                # torch.Size([7, 3, 256, 128])
                gallery_img = torch.cat(gallery_img, dim=0)
                gallery_img = gallery_img.to(device)
                gallery_feats = reidModel(gallery_img)  # torch.Size([7, 2048])
                print("The gallery feature is normalized")
                gallery_feats = torch.nn.functional.normalize(
                    gallery_feats, dim=1, p=2)  # 计算出查询图片的特征向量

                # m: 2
                # n: 7
                m, n = query_feats.shape[0], gallery_feats.shape[0]
                distmat = torch.pow(query_feats, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                    torch.pow(gallery_feats, 2).sum(
                        dim=1, keepdim=True).expand(n, m).t()
                # out=(beta∗M)+(alpha∗mat1@mat2)
                # qf^2 + gf^2 - 2 * qf@gf.t()
                # distmat - 2 * qf@gf.t()
                # distmat: qf^2 + gf^2
                # qf: torch.Size([2, 2048])
                # gf: torch.Size([7, 2048])
                # distmat.addmm_(query_feats, gallery_feats.t(), 1, -2)
                distmat.addmm_(1, -2, query_feats, gallery_feats.t())
                # distmat = (qf - gf)^2
                # distmat = np.array([[1.79536, 2.00926, 0.52790, 1.98851, 2.15138, 1.75929, 1.99410],
                #                     [1.78843, 1.96036, 0.53674, 1.98929, 1.99490, 1.84878, 1.98575]])
                distmat = distmat.cpu().detach().numpy()  # <class 'tuple'>: (3, 12)
                # 平均一下query中同一行人的多个结果
                distmat = distmat.sum(axis=0) / len(query_feats)
                index = distmat.argmin()
                if distmat[index] < dist_thres:
                    print('距离：%s' % distmat[index])
                    plot_one_box(gallery_loc[index], im0,
                                 label='find!', color=colorss[0])
                    # print(gallery_loc[index])
                    zuobiao = gallery_loc[index]
                    xfind = zuobiao[0] + (zuobiao[2] - zuobiao[0]) / 2
                    yfind = zuobiao[3]

                    # print(zuobiao[0]+(zuobiao[2]-zuobiao[0])/2)
                    # print(zuobiao[3])
                    xreal, yreal = goto.celiang2(xfind, yfind)

                    print("真实坐标")
                    print(xreal)
                    print(yreal)

                    if count % 20 == 0:
                        filename = './xfind2.txt'
                        with open(filename, 'a') as file_object:
                            file_object.write(str(xfind))
                            file_object.write("\n")
                        filename1 = './yfind2.txt'
                        with open(filename1, 'a') as file_object:
                            file_object.write(str(yfind))
                            file_object.write("\n")
                        filename2 = './xreal2.txt'
                        with open(filename2, 'a') as file_object:
                            file_object.write(str(xreal))
                            file_object.write("\n")
                        filename3 = './yreal2.txt'
                        with open(filename3, 'a') as file_object:
                            file_object.write(str(yreal))
                            file_object.write("\n")

                    count = count + 1
                    print(count)
                    danger_query = []
                    danger_query_img = []

                    # gallery_loc1 = np.array(gallery_loc)
                    for iss in gallery_loc:
                        zuobiao1 = list(iss)
                        x1 = zuobiao1[0] + (zuobiao1[2] - zuobiao1[0]) / 2
                        y1 = zuobiao1[3]
                        x1real, y1real = goto.celiang2(x1, y1)
                        if (((x1real - xreal) * (x1real - xreal) + (y1real - yreal) * (
                                y1real - yreal)) < 1) and x1real != xreal and y1real != yreal:
                            if danger_first:
                                danger_person = save_one_box(zuobiao1, imc,
                                                             file=save_dir / 'crops' /
                                                             names[c] /
                                                             f'{p.stem}.jpg',
                                                             BGR=True, save=True)  # 保存
                                save_one_img(danger_person, file='app/static/img/danger/' + f'1.jpg')  # 保存
                                danger_person = Image.fromarray(
                                    cv2.cvtColor(danger_person, cv2.COLOR_BGR2RGB))  # PIL: (233, 602)
                                danger_person = build_transforms(reidCfg)(danger_person).unsqueeze(
                                    0)  # torch.Size([1, 3, 256, 128])
                                danger_img.append(danger_person)
                                danger_nums.append(1)
                                
                                # global danger_first
                                danger_first = False
                            else:
                                danger_person = save_one_box(zuobiao1, imc,
                                                             file=save_dir / 'crops' /
                                                             names[c] /
                                                             f'{p.stem}.jpg',
                                                             BGR=True, save=False)
                                danger_query_img.append(danger_person)
                                danger_person = Image.fromarray(
                                    cv2.cvtColor(danger_person, cv2.COLOR_BGR2RGB))  # PIL: (233, 602)
                                danger_person = build_transforms(reidCfg)(danger_person).unsqueeze(
                                    0)  # torch.Size([1, 3, 256, 128])
                                danger_query.append(danger_person)
                                danger_second = False

                            plot_one_box(
                                zuobiao1, im0, label='danger!', color=colorss[1])
                            # path = 'app/static/img/danger/'  # 输入文件夹地址
                            # print(zuobiao1)
                            # x1, y1, x2, y2 = zuobiao1
                            # crop_im = imc[y1:y2, x1:x2]
                            # save_path = path + str(danger_num).zfill(4) + '.jpg'
                            # print(save_path)
                            # cv2.imwrite(save_path, crop_im)
                            # danger_num += 1

                            cv2.line(im0, (int(x1), int(y1)),
                                     (int(xfind), int(yfind)), (0, 0, 255), 2)
                    if not danger_second and danger_query and danger_img:
                        # torch.Size([7, 3, 256, 128])
                        danger_query = torch.cat(danger_query, dim=0)
                        danger_query = danger_query.to(device)
                        danger_query_feats = reidModel(
                            danger_query)  # torch.Size([7, 2048])
                        # torch.Size([7, 3, 256, 128])
                        danger_gallery = torch.cat(danger_img, dim=0)
                        danger_gallery = danger_gallery.to(device)
                        danger_gallery_feats = reidModel(
                            danger_gallery)  # torch.Size([7, 2048])

                        danger_query_feats = torch.nn.functional.normalize(danger_query_feats, dim=1,
                                                                           p=2)  # 计算出查询图片的特征向量
                        danger_gallery_feats = torch.nn.functional.normalize(danger_gallery_feats, dim=1,
                                                                             p=2)  # 计算出查询图片的特征向量
                        print("The danger feature is normalized")
                        # m: 2
                        # n: 7
                        m, n = danger_query_feats.shape[0], danger_gallery_feats.shape[0]
                        danger_distmat = torch.pow(danger_query_feats, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                            torch.pow(danger_gallery_feats, 2).sum(dim=1, keepdim=True).expand(n,
                                                                                               m).t()

                        danger_distmat.addmm_(
                            1, -2, danger_query_feats, danger_gallery_feats.t())
                        danger_distmat = danger_distmat.cpu().detach().numpy()  # <class 'tuple'>: (3, 12)
                        danger_distmat = danger_distmat.sum(
                            axis=0) / len(danger_gallery_feats)  # 平均一下query中同一行人的多个结果
                        for index, danger_dist in enumerate(danger_distmat):
                            if danger_dist > dist_thres:
                                danger_person_tmp = Image.fromarray(
                                    cv2.cvtColor(danger_query_img[index], cv2.COLOR_BGR2RGB))  # PIL: (233, 602)
                                danger_person_tmp = build_transforms(reidCfg)(danger_person_tmp).unsqueeze(
                                    0)
                                # global danger_img
                                danger_img.append(danger_person_tmp)
                                # danger_gallery.append(danger_query[index])
                                save_one_img(danger_query_img[index],
                                             file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg')
                                save_one_img(danger_query_img[index],
                                             file='app/static/img/danger/' + f'{len(danger_img)}.jpg')
                            else:
                                if len(danger_gallery) - index > len(danger_query):
                                    continue
                                while index >= len(danger_nums):
                                    danger_nums.append(0)
                                danger_nums[index] += 1
                               


                    # cv2.imshow('person search', im0)
                    # cv2.waitKey()

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

        # Stream results
        im0 = annotator.result()
        # if view_img:
        #     cv2.imshow(str(p), im0)
        #     cv2.waitKey(1)  # 1 millisecond

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:  # 'video' or 'stream'
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        # release previous video writer
                        vid_writer[i].release()
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer[i] = cv2.VideoWriter(
                        save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)
    return danger_img, danger_first, danger_second, im0, count, danger_nums


if __name__ == '__main__':
    device, half, save_dir, model, dataset, names, save_img, \
        reidModel, query_feats, colorss, vid_path, vid_writer= init_person_reid()
    count = 1
    danger_img = []
    danger_first = True
    danger_second = True
    danger_nums = []
    for index in range(dataset.frames):
        danger_img, danger_first, danger_second, _, count, danger_nums = process_a_img(index, count, device, half, save_dir, model, dataset, names, save_img,
                      reidModel, query_feats, colorss, vid_path, vid_writer, danger_img, danger_first, danger_second, danger_nums)
        print(danger_nums)
