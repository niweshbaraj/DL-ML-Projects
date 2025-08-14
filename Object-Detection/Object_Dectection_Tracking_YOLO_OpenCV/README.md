# Object Detection, Tracking & Segmentation

Various computer vision tasks using YOLO and OpenCV including object detection, tracking, counting, and segmentation.

## Features

- **Multi-Object Detection**: Detect multiple object types simultaneously
- **Object Counting**: Count unique objects (bottles) with tracking IDs
- **People Tracking with Trails**: Track people movement with visual trails
- **Live Camera Feed**: Real-time object detection from webcam
- **Instance Segmentation**: Precise object boundary detection with tracking


## Installation

1. **Clone/Download the project**
   ```bash
   git clone <repository-url>
   cd Object_Detection_Using_YOLO_OpenCV
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv yolo-ocv
   
   # Windows
   yolo-ocv\Scripts\activate
   
   # macOS/Linux
   source yolo-ocv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv8 weights** (automatic on first run)
   - `yolov8n.pt` - Detection model
   - `yolov8n-seg.pt` - Segmentation model

## Usage

### Basic Object Detection
```bash
python simple_object_detection.py
```

### Multi-Object Detection (with video saving)
```bash
python multi_object_from_video.py
```

### Object Counting
```bash
python object_counting.py
```

### People Tracking with Trails
```bash
python people_with_trail.py
```

### Live Camera Feed
```bash
python live_camera_feed.py
```

### Instance Segmentation
```bash
python segmentation.py
```

## Key Features Explained

### Object Counting
- Tracks unique bottle instances across video frames
- Maintains persistent IDs for accurate counting
- Displays real-time count overlay

### Movement Trails
- Records object center positions over time
- Draws colored trails showing movement paths
- Configurable trail length (default: 30 frames)

### Instance Segmentation
- Provides pixel-perfect object boundaries
- More accurate than bounding boxes
- Useful for precise object analysis

## YOLO Classes Used

- **Class 0**: Person
- **Class 1**: Bicycle
- **Class 2**: Car
- **Class 39**: Bottle
- **All Classes**: Multi-object detection (80+ COCO classes)

## Controls

- **'q'**: Quit/Exit video playback
- Videos are automatically saved to `data/annotated_data/videos/`

## Customization

### Change Detection Classes
```python
# For specific classes
results = model.track(frame, classes=[0, 2], persist=True)  # Person + Car

# For all classes
results = model.track(frame, persist=True)  # All 80 COCO classes
```

### Adjust Trail Length
```python
trail = defaultdict(lambda: deque(maxlen=50))  # Longer trails
```

### Modify Video Input
```python
video_path = os.path.join(cwd, "data/raw_data/videos/your_video.mp4")
```

## Requirements Details

```
opencv-python>=4.5.0
ultralytics>=8.0.0
numpy>=1.21.0
```
