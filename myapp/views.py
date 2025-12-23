from django.shortcuts import render
import os
from django.conf import settings

# Create your views here.
import time
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from django.shortcuts import render
from .camera import controller
from .camera import _pos_x_q, _pos_y_q, _scale_q
from django.views.decorators.csrf import csrf_exempt

def index(request):
    if not controller.running:
        controller.start()
        time.sleep(0.1)

    asset_dir = os.path.join(settings.BASE_DIR,'myapp', 'static', 'assets','he')
    clothes = [
        f for f in os.listdir(asset_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    ]

    return render(request, "index.html", {
        "clothes": sorted(clothes)
    })


def mjpeg_stream(request):
    def generate():
        while controller.running:
            frame = controller.get_frame()
            if frame:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            else:
                time.sleep(0.05)

    return StreamingHttpResponse(
        generate(),
        content_type="multipart/x-mixed-replace; boundary=frame"
    )

def next_cloth(request):
    controller.next_cloth()
    return JsonResponse({"status":"ok", "cur_id": controller.cur_id})

def prev_cloth(request):
    controller.prev_cloth()
    return JsonResponse({"status":"ok", "cur_id": controller.cur_id})

def select_cloth(request, idx):
    controller.set_cloth(int(idx))
    return JsonResponse({"status": "ok", "cur_id": controller.cur_id})


def homepage(request):
    return render(request, "home.html")


def start_camera(request):
    if not controller.running:
        controller.start()
        time.sleep(0.1)
    return JsonResponse({"status": "camera started"})

@csrf_exempt
def stop_camera(request):
    controller.stop()
    return JsonResponse({"status": "camera stopped"})


def select_cloth(request, idx):
    if controller.num == 0:
        return JsonResponse({"status": "empty", "cur_id": 0})

    idx = max(0, min(idx, controller.num - 1))

    controller.cur_id = idx

    # IMPORTANT: reset smoothing buffers
    _pos_x_q.clear()
    _pos_y_q.clear()
    _scale_q.clear()

    return JsonResponse({
        "status": "ok",
        "cur_id": controller.cur_id
    })

def howitworks(request):
    return render(request,"howitworks.html")

def about(request):
    return render(request,"about.html")

def feature(request):
    return render(request,"feature.html")
