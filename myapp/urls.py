from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.homepage, name='home'),
    path("index/", views.index, name="index"),
    path("stream/", views.mjpeg_stream, name="stream"),
    path("next/", views.next_cloth, name="next"),
    path("prev/", views.prev_cloth, name="prev"),
    path("stop/", views.stop_camera, name="stop"),
    path("select/<int:idx>/", views.select_cloth),
    path("howitworks/", views.howitworks,name="howitworks"),
    path("camera/start/", views.start_camera),
    path("camera/stop/", views.stop_camera),
    path("about/", views.about, name='about'),
    path("feature/",views.feature,name='feature'),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])