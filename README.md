# AirCanvas 🎨

**Draw in the air with your hands.**

AirCanvas is a real-time virtual drawing application built with **Python**, **OpenCV**, and **MediaPipe**. It uses a webcam to track hand landmarks, allowing you to draw, create geometric shapes, choose colors, and adjust brush thickness without touching a mouse or touchscreen.

## Features

- Real-time hand tracking with MediaPipe Hand Landmarker
- Freehand drawing with your left hand
- Shape tools: **Circle**, **Square**, **Triangle**, and **Rectangle**
- Eight colors, including black
- On-screen toolbar controlled by your right index finger
- Gesture-controlled brush thickness (2–40 px) with a live indicator
- Live shape previews before placing them on the canvas
- Clear and save drawings using keyboard shortcuts
- Mirrored camera view, hand landmark visualization, and FPS display

## How to Use

| Input | Action |
| --- | --- |
| Left hand: separate thumb and index finger | Start drawing or adjusting a shape |
| Left hand: bring thumb and index finger together | Stop drawing or commit a shape |
| Right index finger: hover over a toolbar button | Select a color or tool after a short hold |
| Right hand: pinch thumb and index finger, then move up | Increase brush thickness |
| Right hand: pinch thumb and index finger, then move down | Decrease brush thickness |
| Right hand: release pinch | Stop adjusting thickness |
| `C` | Clear the canvas |
| `S` | Save the drawing as a PNG |
| `Q` | Quit |

**Note:** Left and right refer to your physical hands when the camera is mirrored (`MIRROR_CAMERA = True`). Gesture recognition can vary with lighting, camera angle, and hand visibility.

## Requirements

- Python 3.10 or newer (use a version supported by your installed MediaPipe release)
- A working webcam
- The Python packages listed in `requirements.txt`
- MediaPipe's `hand_landmarker.task` model file

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ParsaPouramiri8/AirCanvas.git
   cd AirCanvas
   ```

2. Create and activate a virtual environment (recommended):

   **Windows (PowerShell):**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS / Linux:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the **Hand Landmarker** model from the [MediaPipe Hand Landmarker documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) and save it as `models/hand_landmarker.task`.

5. Check your camera settings in `aircanvas.py`. The default `CAMERA_INDEX = 1` is specific to the original development setup; try `0` if your webcam is not detected.

6. Run the application:

   ```bash
   python aircanvas.py
   ```

## Project Structure

```text
AirCanvas/
├── aircanvas.py               # Main application
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git exclusions
├── LICENSE                    # Project license
└── models/
    └── hand_landmarker.task   # MediaPipe model (download separately)
```

The application expects the model path to be configured as:

```python
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
```

## Configuration

You can customize these values near the top of `aircanvas.py`:

| Setting | Default | Description |
| --- | --- | --- |
| `CAMERA_INDEX` | `1` | Camera device index |
| `MIRROR_CAMERA` | `True` | Mirror the webcam image |
| `BRUSH_THICKNESS` | `7` | Initial brush size in pixels |
| `MIN_THICKNESS` | `2` | Minimum brush size |
| `MAX_THICKNESS` | `40` | Maximum brush size |
| `BUTTON_HOLD_SECONDS` | `0.45` | Hover duration for selecting a toolbar item |
| `PINCH_CONTROL_THRESHOLD` | `0.40` | Right-hand pinch detection threshold |
| `PINCH_PIXELS_PER_STEP` | `12` | Vertical movement needed for a thickness step |

## How It Works

1. **OpenCV** captures frames from the webcam.
2. **MediaPipe Hand Landmarker** detects hand landmarks and identifies the left and right hands.
3. The left-hand thumb–index distance controls drawing, while the right hand controls the toolbar and brush thickness.
4. Drawings are stored on a separate canvas and composited over the live camera feed using a mask, so black strokes remain visible.
5. Shape tools show a live preview before the shape is committed.

## Screenshots and Demo

Screenshots or a short demonstration GIF can be added here after recording the application.

## Known Limitations

- Tracking accuracy depends on lighting, camera quality, and hand visibility.
- The interface is currently built with OpenCV rather than a standalone desktop GUI.
- Saved PNGs contain the drawing canvas on a black background, not the live webcam image.
- Camera index and camera backend settings may need adjustment on other systems.

## License

See the [`LICENSE`](LICENSE) file for license terms. If you select the MIT License when creating the GitHub repository, GitHub will generate that file for you.
