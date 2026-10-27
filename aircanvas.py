import cv2
import mediapipe as mp
import numpy as np
import time
import math
from pathlib import Path

# AirCanvas v1.0
# -----------------------------------------------------------------------------
# A webcam-based, gesture-controlled drawing application built with OpenCV and
# MediaPipe Hand Landmarker.
#
# Controls:
#   Left hand:  separate thumb/index to draw; pinch them together to stop.
#   Right hand: hover over toolbar buttons to select a color or drawing tool.
#               Pinch thumb/index and move vertically to adjust brush size.
#   Keyboard:   C = clear, S = save canvas, Q = quit.
#
# Requirements: opencv-python, mediapipe, numpy, hand_landmarker.task
# The model file must be in the same directory as this script.
# -----------------------------------------------------------------------------

# CAMERA SETTINGS
CAMERA_INDEX = 1
MIRROR_CAMERA = True

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

DETECTION_WIDTH = 640
DETECTION_HEIGHT = 360

# BRUSH SETTINGS
BRUSH_THICKNESS = 7
MIN_THICKNESS = 2
MAX_THICKNESS = 40

SMOOTHING = 0.45

# LEFT HAND DRAWING GESTURE
PINCH_START_DRAW = 0.55
PINCH_STOP_DRAW = 0.35

# RIGHT HAND THICKNESS GESTURE
PINCH_CONTROL_THRESHOLD = 0.40
PINCH_PIXELS_PER_STEP = 12

# TOOLBAR
BUTTON_HOLD_SECONDS = 0.45

# COLORS (BGR)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
DARK = (30, 30, 30)

PALETTE = [
    ("Blue", (255, 0, 0)),
    ("Red", (0, 0, 255)),
    ("Green", (0, 255, 0)),
    ("Yellow", (0, 255, 255)),
    ("Purple", (255, 0, 170)),
    ("White", (255, 255, 255)),
    ("Orange", (0, 165, 255)),
    ("Black", (0, 0, 0)),
]

TOOLS = [
    "Brush",
    "Circle",
    "Square",
    "Triangle",
    "Rectangle"
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def distance(a, b):
    """Return the Euclidean distance between two normalized landmarks."""
    return math.hypot(
        a.x - b.x,
        a.y - b.y
    )


def point_from_landmark(landmark, width, height):
    """Convert a normalized hand landmark to pixel coordinates."""
    return (
        int(landmark.x * width),
        int(landmark.y * height)
    )


def get_hand_label(result, index):
    """Read the detected handedness and account for camera mirroring."""
    if index >= len(result.handedness):
        return "Unknown"

    label = result.handedness[index][0].category_name

    # MediaPipe handedness assumes a selfie-style (mirrored) image.
    if not MIRROR_CAMERA:
        if label == "Left":
            return "Right"
        if label == "Right":
            return "Left"

    return label


def draw_hand_skeleton(frame, landmarks, width, height):
    """Overlay the 21 hand landmarks and their connections on a frame."""
    for start, end in HAND_CONNECTIONS:
        p1 = point_from_landmark(
            landmarks[start], width, height
        )
        p2 = point_from_landmark(
            landmarks[end], width, height
        )

        cv2.line(frame, p1, p2, RED, 2)

    for index, landmark in enumerate(landmarks):
        point = point_from_landmark(
            landmark, width, height
        )

        color = BLUE if index == 8 else GREEN
        radius = 10 if index == 8 else 3

        cv2.circle(
            frame,
            point,
            radius,
            color,
            -1
        )


# ==========================================
# TOOLBAR FUNCTIONS
# ==========================================

def get_toolbar_buttons(width):
    """Build color and tool hitboxes in the upper-left corner."""
    buttons = []

    color_width = 60
    color_height = 48
    color_gap = 7

    start_x = 15
    color_y = 15

    # First row: color swatches, including black.
    for i, (name, color) in enumerate(PALETTE):
        x1 = start_x + i * (
            color_width + color_gap
        )

        buttons.append({
            "type": "color",
            "index": i,
            "label": name,
            "rect": (
                x1,
                color_y,
                x1 + color_width,
                color_y + color_height
            )
        })

    # Second row: freehand brush and four geometric shapes.
    tool_width = 94
    tool_height = 52
    tool_gap = 8
    tool_y = 76

    for i, tool in enumerate(TOOLS):
        x1 = start_x + i * (
            tool_width + tool_gap
        )

        buttons.append({
            "type": "tool",
            "index": i,
            "label": tool,
            "rect": (
                x1,
                tool_y,
                x1 + tool_width,
                tool_y + tool_height
            )
        })

    return buttons


def get_hovered_button(buttons, x, y):
    """Return the toolbar button under a fingertip, or None."""
    for button_id, button in enumerate(buttons):
        x1, y1, x2, y2 = button["rect"]

        if (
            x1 <= x <= x2
            and y1 <= y <= y2
        ):
            return button_id

    return None


def draw_toolbar(
    frame,
    buttons,
    selected_color_index,
    selected_tool_index,
    hover_button_id,
    hover_progress
):
    """Render the toolbar, active selections, and hover progress bars."""
    for button_id, button in enumerate(buttons):
        x1, y1, x2, y2 = button["rect"]

        selected = (
            button["type"] == "color"
            and button["index"] == selected_color_index
        ) or (
            button["type"] == "tool"
            and button["index"] == selected_tool_index
        )

        if button["type"] == "color":
            fill = PALETTE[button["index"]][1]
        else:
            fill = (65, 55, 45)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            fill,
            -1
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            YELLOW if selected else WHITE,
            3 if selected else 1
        )

        if button["type"] == "color":
            cv2.rectangle(
                frame,
                (x1, y2 - 19),
                (x2, y2),
                DARK,
                -1
            )

        cv2.putText(
            frame,
            button["label"],
            (x1 + 5, y2 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39 if button["type"] == "color" else 0.47,
            WHITE,
            1
        )

        if button_id == hover_button_id:
            progress_width = int(
                (x2 - x1) * hover_progress
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x1 + progress_width, y1 + 5),
                YELLOW,
                -1
            )


# ==========================================
# SHAPE DRAWING
# ==========================================

def draw_shape(
    target,
    tool,
    start_point,
    end_point,
    color,
    thickness
):
    """Draw an outlined geometric shape onto an image or grayscale mask."""
    if start_point is None or end_point is None:
        return

    x1, y1 = start_point
    x2, y2 = end_point

    if tool == "Circle":
        radius = int(
            math.hypot(
                x2 - x1,
                y2 - y1
            )
        )

        if radius > 0:
            cv2.circle(
                target,
                start_point,
                radius,
                color,
                thickness,
                cv2.LINE_AA
            )

    elif tool == "Rectangle":
        cv2.rectangle(
            target,
            start_point,
            end_point,
            color,
            thickness,
            cv2.LINE_AA
        )

    elif tool == "Square":
        dx = x2 - x1
        dy = y2 - y1

        side = min(
            abs(dx),
            abs(dy)
        )

        end_x = x1 + (
            side if dx >= 0 else -side
        )

        end_y = y1 + (
            side if dy >= 0 else -side
        )

        cv2.rectangle(
            target,
            start_point,
            (end_x, end_y),
            color,
            thickness,
            cv2.LINE_AA
        )

    elif tool == "Triangle":
        top = min(y1, y2)
        bottom = max(y1, y2)
        left = min(x1, x2)
        right = max(x1, x2)

        points = np.array([
            [(left + right) // 2, top],
            [left, bottom],
            [right, bottom]
        ], dtype=np.int32)

        cv2.polylines(
            target,
            [points],
            True,
            color,
            thickness,
            cv2.LINE_AA
        )


# ==========================================
# CANVAS COMPOSITING
# ==========================================

def combine_canvas(frame, canvas, drawn_mask):
    """Overlay committed strokes onto the live camera frame using a mask."""
    """
    A dedicated stroke mask keeps black paint visible on the camera feed.
    """
    background = cv2.bitwise_and(
        frame,
        frame,
        mask=cv2.bitwise_not(drawn_mask)
    )

    foreground = cv2.bitwise_and(
        canvas,
        canvas,
        mask=drawn_mask
    )

    return cv2.add(
        background,
        foreground
    )


# ==========================================
# THICKNESS BAR
# ==========================================

def draw_thickness_bar(frame, thickness):
    """Display the current brush width as a labeled progress bar."""
    x1 = 20
    y1 = 315

    bar_width = 260
    bar_height = 20

    progress = (
        (thickness - MIN_THICKNESS)
        / (MAX_THICKNESS - MIN_THICKNESS)
    )

    cv2.putText(
        frame,
        f"Thickness: {thickness} px",
        (x1, y1 - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        WHITE,
        2
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x1 + bar_width, y1 + bar_height),
        (65, 65, 65),
        -1
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (
            x1 + int(bar_width * progress),
            y1 + bar_height
        ),
        (255, 190, 60),
        -1
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x1 + bar_width, y1 + bar_height),
        WHITE,
        1
    )


# ==========================================
# CAMERA SETUP
# ==========================================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

cap.set(
    cv2.CAP_PROP_FPS,
    30
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)

if not cap.isOpened():
    raise RuntimeError("Could not open camera!")


# ==========================================
# MEDIAPIPE SETUP
# ==========================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"

if not MODEL_PATH.exists():
    cap.release()
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# APPLICATION STATE
# ==========================================

canvas = None
drawn_mask = None

selected_color_index = 0
selected_tool_index = 0

brush_thickness = BRUSH_THICKNESS

drawing_active = False
previous_point = None
smoothed_point = None

shape_start = None
shape_end = None

hover_button_id = None
hover_start_time = None
hover_progress = 0.0

# Right-hand pinch state
pinch_previous_y = None
pinch_movement_accumulator = 0.0

start_time = time.monotonic()
last_timestamp = -1

previous_time = time.perf_counter()
fps = 0.0


# ==========================================
# MAIN LOOP
# ==========================================

try:
    with HandLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:

            ret, frame = cap.read()

            if not ret:
                print("Camera frame unavailable.")
                break

            # Mirror camera
            if MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]

            # Initialize canvas
            if canvas is None or canvas.shape != frame.shape:
                canvas = np.zeros_like(frame)

                drawn_mask = np.zeros(
                    (height, width),
                    dtype=np.uint8
                )

                drawing_active = False
                previous_point = None
                smoothed_point = None
                shape_start = None
                shape_end = None

            buttons = get_toolbar_buttons(width)

            # ==================================
            # HAND DETECTION
            # ==================================

            small_frame = cv2.resize(
                frame,
                (DETECTION_WIDTH, DETECTION_HEIGHT),
                interpolation=cv2.INTER_AREA
            )

            rgb_frame = cv2.cvtColor(
                small_frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            timestamp_ms = max(
                timestamp_ms,
                last_timestamp + 1
            )

            last_timestamp = timestamp_ms

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            hand_count = len(result.hand_landmarks)

            left_hand_found = False
            right_hand_found = False

            current_hover_button = None

            # ==================================
            # PROCESS HANDS
            # ==================================

            for hand_index, landmarks in enumerate(
                result.hand_landmarks
            ):

                hand_label = get_hand_label(
                    result,
                    hand_index
                )

                draw_hand_skeleton(
                    frame,
                    landmarks,
                    width,
                    height
                )

                wrist_x, wrist_y = point_from_landmark(
                    landmarks[0],
                    width,
                    height
                )

                cv2.putText(
                    frame,
                    hand_label,
                    (
                        max(0, wrist_x - 30),
                        max(30, wrist_y + 35)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    YELLOW,
                    2
                )

                # ==============================
                # LEFT HAND: DRAWING
                # ==============================

                if hand_label == "Left":

                    left_hand_found = True

                    index_tip = landmarks[8]
                    thumb_tip = landmarks[4]

                    palm_size = max(
                        distance(
                            landmarks[0],
                            landmarks[9]
                        ),
                        0.0001
                    )

                    pinch_ratio = (
                        distance(index_tip, thumb_tip)
                        / palm_size
                    )

                    was_drawing = drawing_active

                    # Use two thresholds (hysteresis) to prevent flickering
                    # between draw and pause when fingers are near the limit.
                    if drawing_active:
                        if pinch_ratio < PINCH_STOP_DRAW:
                            drawing_active = False
                    else:
                        if pinch_ratio > PINCH_START_DRAW:
                            drawing_active = True

                    # Finger position
                    raw_x = index_tip.x * width
                    raw_y = index_tip.y * height

                    # Exponential smoothing reduces fingertip tracking jitter.
                    if smoothed_point is None:
                        smoothed_point = (
                            raw_x,
                            raw_y
                        )
                    else:
                        old_x, old_y = smoothed_point

                        smoothed_point = (
                            old_x + SMOOTHING * (raw_x - old_x),
                            old_y + SMOOTHING * (raw_y - old_y)
                        )

                    current_point = (
                        int(smoothed_point[0]),
                        int(smoothed_point[1])
                    )

                    selected_tool = TOOLS[
                        selected_tool_index
                    ]

                    selected_color = PALETTE[
                        selected_color_index
                    ][1]

                    # Capture the first point when a new stroke begins.
                    if drawing_active and not was_drawing:
                        previous_point = None

                        if selected_tool != "Brush":
                            shape_start = current_point
                            shape_end = current_point

                    # Brush strokes are committed immediately; geometric
                    # shapes update their endpoints until the gesture ends.
                    if drawing_active:

                        if selected_tool == "Brush":

                            if previous_point is not None:
                                cv2.line(
                                    canvas,
                                    previous_point,
                                    current_point,
                                    selected_color,
                                    brush_thickness,
                                    cv2.LINE_AA
                                )

                                cv2.line(
                                    drawn_mask,
                                    previous_point,
                                    current_point,
                                    255,
                                    brush_thickness,
                                    cv2.LINE_AA
                                )

                            else:
                                radius = max(
                                    1,
                                    brush_thickness // 2
                                )

                                cv2.circle(
                                    canvas,
                                    current_point,
                                    radius,
                                    selected_color,
                                    -1,
                                    cv2.LINE_AA
                                )

                                cv2.circle(
                                    drawn_mask,
                                    current_point,
                                    radius,
                                    255,
                                    -1,
                                    cv2.LINE_AA
                                )

                            previous_point = current_point

                        else:
                            shape_end = current_point

                    # Commit a completed shape to both the color canvas
                    # and the visibility mask when drawing stops.
                    elif was_drawing:

                        if (
                            selected_tool != "Brush"
                            and shape_start is not None
                            and shape_end is not None
                        ):
                            draw_shape(
                                canvas,
                                selected_tool,
                                shape_start,
                                shape_end,
                                selected_color,
                                brush_thickness
                            )

                            draw_shape(
                                drawn_mask,
                                selected_tool,
                                shape_start,
                                shape_end,
                                255,
                                brush_thickness
                            )

                        previous_point = None
                        shape_start = None
                        shape_end = None

                    else:
                        previous_point = None

                # ==============================
                # RIGHT HAND: CONTROLS
                # ==============================

                elif hand_label == "Right":

                    right_hand_found = True

                    thumb_tip = landmarks[4]
                    index_tip = landmarks[8]

                    palm_size = max(
                        distance(
                            landmarks[0],
                            landmarks[9]
                        ),
                        0.0001
                    )

                    pinch_ratio = (
                        distance(
                            thumb_tip,
                            index_tip
                        )
                        / palm_size
                    )

                    # --------------------------
                    # PINCH: CHANGE THICKNESS
                    # --------------------------

                    if pinch_ratio < PINCH_CONTROL_THRESHOLD:

                        pinch_y = (
                            (
                                thumb_tip.y
                                + index_tip.y
                            ) / 2
                        ) * height

                        if pinch_previous_y is None:
                            pinch_previous_y = pinch_y
                            pinch_movement_accumulator = 0.0

                        else:
                            delta_y = (
                                pinch_previous_y
                                - pinch_y
                            )

                            pinch_previous_y = pinch_y

                            # Ignore tiny vertical movements caused by tracking noise.
                            if abs(delta_y) > 1.5:
                                pinch_movement_accumulator += delta_y

                            steps = int(
                                pinch_movement_accumulator
                                / PINCH_PIXELS_PER_STEP
                            )

                            if steps != 0:
                                brush_thickness = max(
                                    MIN_THICKNESS,
                                    min(
                                        MAX_THICKNESS,
                                        brush_thickness + steps
                                    )
                                )

                                pinch_movement_accumulator -= (
                                    steps * PINCH_PIXELS_PER_STEP
                                )

                        # Disable toolbar selection during brush-size adjustment.
                        current_hover_button = None

                    # --------------------------
                    # NORMAL: SELECT BUTTONS
                    # --------------------------

                    else:
                        pinch_previous_y = None
                        pinch_movement_accumulator = 0.0

                        finger_x, finger_y = (
                            point_from_landmark(
                                landmarks[8],
                                width,
                                height
                            )
                        )

                        current_hover_button = (
                            get_hovered_button(
                                buttons,
                                finger_x,
                                finger_y
                            )
                        )

            # ==================================
            # RESET LOST HAND STATES
            # ==================================

            if not left_hand_found:
                drawing_active = False
                previous_point = None
                smoothed_point = None
                shape_start = None
                shape_end = None

            if not right_hand_found:
                pinch_previous_y = None
                pinch_movement_accumulator = 0.0

            # ==================================
            # TOOLBAR SELECTION TIMER
            # ==================================

            now = time.monotonic()

            if current_hover_button is None:
                hover_button_id = None
                hover_start_time = None
                hover_progress = 0.0

            elif current_hover_button != hover_button_id:
                hover_button_id = current_hover_button
                hover_start_time = now
                hover_progress = 0.0

            else:
                elapsed = now - hover_start_time

                hover_progress = min(
                    elapsed / BUTTON_HOLD_SECONDS,
                    1.0
                )

                if elapsed >= BUTTON_HOLD_SECONDS:
                    button = buttons[hover_button_id]

                    if button["type"] == "color":
                        selected_color_index = button["index"]

                    elif button["type"] == "tool":
                        if selected_tool_index != button["index"]:
                            selected_tool_index = button["index"]

                            drawing_active = False
                            previous_point = None
                            shape_start = None
                            shape_end = None

            # ==================================
            # COMBINE CAMERA AND CANVAS
            # ==================================

            output = combine_canvas(
                frame,
                canvas,
                drawn_mask
            )

            selected_tool = TOOLS[
                selected_tool_index
            ]

            # Preview shapes on the displayed frame without committing them.
            if (
                drawing_active
                and selected_tool != "Brush"
                and shape_start is not None
                and shape_end is not None
            ):
                draw_shape(
                    output,
                    selected_tool,
                    shape_start,
                    shape_end,
                    PALETTE[selected_color_index][1],
                    brush_thickness
                )

            # Toolbar
            draw_toolbar(
                output,
                buttons,
                selected_color_index,
                selected_tool_index,
                hover_button_id,
                hover_progress
            )

            # ==================================
            # FPS
            # ==================================

            current_time = time.perf_counter()
            delta = current_time - previous_time
            previous_time = current_time

            if delta > 0:
                fps = (
                    0.9 * fps
                    + 0.1 * (1.0 / delta)
                )

            mode = (
                "DRAW"
                if drawing_active
                else "HOVER"
            )

            # ==================================
            # HUD
            # ==================================

            cv2.putText(
                output,
                f"FPS: {fps:.1f}",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                GREEN,
                2
            )

            cv2.putText(
                output,
                f"Hands: {hand_count}",
                (20, 205),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                WHITE,
                2
            )

            cv2.putText(
                output,
                f"Mode: {mode}",
                (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                YELLOW,
                2
            )

            cv2.putText(
                output,
                f"Tool: {selected_tool}",
                (20, 275),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                WHITE,
                2
            )

            # Thickness indicator
            draw_thickness_bar(
                output,
                brush_thickness
            )

            # Show a visual hint while the right-hand size gesture is active.
            if pinch_previous_y is not None:
                cv2.putText(
                    output,
                    "SIZE CONTROL ACTIVE",
                    (20, 375),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    YELLOW,
                    2
                )

            cv2.putText(
                output,
                "LEFT: Draw | RIGHT: Select / Pinch: Size | C: Clear | S: Save | Q: Quit",
                (20, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                WHITE,
                2
            )

            cv2.imshow(
                "AirCanvas v1.0",
                output
            )

            # ==================================
            # KEYBOARD
            # ==================================

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("c"):
                canvas[:] = 0
                drawn_mask[:] = 0

                drawing_active = False
                previous_point = None
                shape_start = None
                shape_end = None

            elif key == ord("s"):
                filename = time.strftime(
                    "aircanvas_%Y%m%d_%H%M%S.png"
                )

                cv2.imwrite(
                    filename,
                    canvas
                )

                print(f"Saved: {filename}")

finally:
    cap.release()
    cv2.destroyAllWindows()