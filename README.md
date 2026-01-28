# VirtuFit — AI Virtual Dressing Room

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-3.2%2B-green)](https://www.djangoproject.com/)

A production-minded, AI-powered virtual dressing room that lets users try on clothes in real time using a webcam. VirtuFit combines pose estimation, hand-gesture interaction, and smooth image blending to deliver a responsive, contact-free virtual try-on experience for e-commerce demos, research, and prototypes.

---

## Key highlights

- Real-time camera-based try-on (no app required — runs in the browser)
- Robust pose detection for accurate overlay placement (MediaPipe)
- Hand gesture controls for intuitive outfit navigation
- Smooth motion stabilization and auto-scaling of clothing assets
- Lightweight Django backend to stream frames and serve the UI

---

## Features

- Real-time webcam feed with transparent clothing overlay
- AI pose detection for shoulder / hip alignment and scaling
- Hand gesture recognition:
  - Swipe left / right → switch outfits
  - Pinch gesture → fast switch
  - Thumbs-up → capture photo
- Motion smoothing to reduce jitter
- Instant outfit capture and download
- Modern, responsive UI (glassmorphism + gradients)

---

## How it works (high level)

1. Webcam frames are captured in the browser and streamed to the backend.
2. MediaPipe Pose and Hand models detect body keypoints and hand landmarks.
3. Clothing images (with transparent backgrounds) are positioned, scaled, and blended on top of the detected body using OpenCV.
4. Motion smoothing is applied to keypoints to produce stable overlays.
5. Hand gestures are interpreted to navigate outfits or capture photos.

---

## Tech stack

- Backend: Python, Django
- Computer vision: OpenCV, MediaPipe, NumPy
- Frontend: HTML5, CSS3, JavaScript (modern responsive UI)
- Optional: GPU acceleration for heavy CV workloads

---

## Quick start (local development)

Prerequisites
- Python 3.8 or later
- pip
- A modern browser (Chrome/Edge/Firefox) with camera access

Clone and set up
```bash
git clone https://github.com/JUAAKASH123/Virtufit-Django.git
cd Virtufit-Django

# create venv and activate
python -m venv .venv
# On macOS / Linux
source .venv/bin/activate
# On Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Environment
- Create a `.env` or set environment variables:
  - DJANGO_SECRET_KEY (or use a default for development)
  - DEBUG=1
  - ALLOWED_HOSTS=localhost,127.0.0.1

Apply migrations and run
```bash
python manage.py migrate
python manage.py runserver
```

Open http://127.0.0.1:8000 in your browser, allow camera permissions, and try the demo.

Notes:
- If MediaPipe or OpenCV installation fails on certain platforms, consult their official installation guides.
- For improved performance you can run on a machine with a dedicated GPU and appropriate acceleration libraries (optional).

---

## Gesture controls (user-facing)

- Swipe left / right (hand movement) — cycle through outfits
- Pinch gesture — fast switch to next outfit
- Thumbs-up — capture and save the current view as an image

These gestures can be tuned or remapped in the frontend gesture handler.

---

## Project structure (overview)

- virtufit/ — Django project files
- app/ — main app (views, templates, static assets)
- static/ — frontend assets (CSS, JS, images)
- models/ — clothing assets and metadata (optional)
- requirements.txt — Python dependencies

---

## Deployment tips

- Use HTTPS to securely access the camera from production domains.
- Configure allowed hosts and a secure SECRET_KEY in production.
- Consider serving static assets from a CDN and using a process manager (gunicorn + nginx) for the Django app.
- If scaling CV work, isolate heavy processing into an asynchronous worker or a dedicated inference service.

---

## Contributing

Contributions are welcome! Suggested workflow:
1. Fork the repository
2. Create a feature branch (feature/short-description)
3. Open a pull request describing your changes
4. Ensure linters and basic tests (if any) pass

Please open issues for feature requests, bugs, or performance problems.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Credits & resources

- MediaPipe — pose and hand tracking
- OpenCV — image processing and blending
- Inspiration: modern virtual-try-on research and interactive demos

---

## Author

Aakash Udai — AI & Full-Stack Developer  
Building the future of virtual fashion 🚀  
GitHub: [JUAAKASH123](https://github.com/JUAAKASH123)

