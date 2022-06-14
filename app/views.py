# coding=utf-8
from django.contrib import messages
from django.shortcuts import render, render_to_response, redirect
from django import forms
from django.http import HttpResponse, FileResponse, JsonResponse, StreamingHttpResponse
# from web.app.models import *
from detect import init_person_reid, process_a_img
import os
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = 'app/static/upload'
FILE_DIR = 'static/upload'
# Create your views here.


def index(request):
    return render(request, 'index.html')


def upload(request):
    if request.method == 'POST':  # 获取对象
        print("request.FILES", request.FILES)
        video = request.FILES.get('video')
        img = request.FILES.get('img')
        if not video or not img:
            messages.success(request, "请选择文件！")
        else:
            f = open(os.path.join(BASE_DIR, UPLOAD_DIR, video.name), 'wb')
            for chunk in video.chunks():
                f.write(chunk)
            f.close()
            f = open(os.path.join(BASE_DIR, UPLOAD_DIR, img.name), 'wb')
            for chunk in img.chunks():
                f.write(chunk)
            f.close()
            messages.success(request, "上传成功！")
            test_video = os.path.join(FILE_DIR, video.name)
            test_img = os.path.join(FILE_DIR, img.name)
            return render(request, "index.html", {'test_video': test_video, 'test_img': test_img})

    return render(request, "index.html")


def upload_video(request):
    if request.method == 'POST':  # 获取对象
        obj = request.FILES.get('video')
        if not obj:
            messages.success(request, "请选择文件！")
        else:
            f = open(os.path.join(BASE_DIR, UPLOAD_DIR, obj.name), 'wb')
            for chunk in obj.chunks():
                f.write(chunk)
            f.close()
            messages.success(request, "上传成功！")
            test_video = os.path.join(FILE_DIR, obj.name)
            return render(request, "index.html", {'test_video': test_video})

    return render(request, "index.html")


def upload_img(request):
    if request.method == 'POST':  # 获取对象
        obj = request.FILES.get('img')
        if not obj:
            messages.success(request, "请选择文件！")
        else:
            f = open(os.path.join(BASE_DIR, UPLOAD_DIR, obj.name), 'wb')
            for chunk in obj.chunks():
                f.write(chunk)
            f.close()
            messages.success(request, "上传成功！")
            test_img = os.path.join(FILE_DIR, obj.name)
            return render(request, "index.html", {'test_img': test_img})

    return render(request, "index.html")


def test(request):
    messages.success(request, "开始处理！")

    return render(request, "index.html")

# @csrf_exempt


def getlen(request):
    path = 'app\static\img\danger'      # 输入文件夹地址
    files = os.listdir(path)
    # imgs_length = len(files)
    res = []
    for file in files:
        res.append(file)
    danger_nums = []
    filename = './danger_time.txt'
    with open(filename, 'r') as file_object:
        nums = file_object.readline()
        nums = nums[1:len(nums)-2]
        # nums = nums.split(',')
        # for num in nums:
        #     danger_nums.append(num)
        # print(danger_nums)
    # print(imgs_length)
    res.append(nums)
    return HttpResponse(res)


def video_feed(request):
    return StreamingHttpResponse(gen(10), content_type="multipart/x-mixed-replace; boundary=frame")


def preview(request):
    test_video = 'static/video/nanyi2.mp4'
    output_video = 'static/video/nanyi2_out.mp4'
    messages.success(request, "开始预览！")
    return render(request, "index.html", {'test_video': test_video, 'output_video': output_video})


def download(request):
    file = open('download/output.mp4', 'rb')
    response = FileResponse(file)
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = 'attachment;filename="output.mp4"'
    return response


def gen(rate=30):
    device, half, save_dir, model, dataset, names, save_img, \
        reidModel, query_feats, colorss, vid_path, vid_writer = init_person_reid()
    count = 1
    danger_img = []
    danger_first = True
    danger_second = True
    danger_nums = []
    for index in range(dataset.frames):
        danger_img, danger_first, danger_second, image, count, danger_nums = process_a_img(index, count, device, half, save_dir, model, dataset, names, save_img,
                                                                              reidModel, query_feats, colorss, vid_path, vid_writer, danger_img, danger_first, danger_second, danger_nums)
        filename = './danger_time.txt'
        with open(filename, 'w') as file_object:
            file_object.write(str(danger_nums))
            file_object.write("\n")
        # ret, jpeg = cv2.imencode('.jpg', image)
        ret, jpeg = cv2.imencode('.JPG', image)
        frame = jpeg.tobytes()
        # return frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

