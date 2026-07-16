import cherrypy
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import json
import threading
import io
from typing import Optional, List, Dict
from owl_detector import Owlv2Detector, OwlViTDetector
from cherrypy.process.plugins import SignalHandler

class ObjectDetectionApp:
    """CherryPy web app for object detection with webcam stream."""
    
    def __init__(self):
        """Initialize the detection app with default settings."""
        # Settings
        self.detector_type = 'owlv2'
        self.model_name = None
        self.objects = ["a person", "human face", "a hand"]
        self.threshold = 0.17
        self.frame_width = 960
        self.frame_height = 540
        self.video_device_id = 0
        self.frame_skip = 5
        
        # State
        self.detector = None
        self.cap = None
        self.last_detections = []
        self.current_frame = None
        self.current_frame_with_detections = None
        self.lock = threading.Lock()
        self.running = False
        self.frame_count = 0
        
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the detector based on current settings."""
        try:
            if self.detector_type == 'owlv2':
                self.detector = Owlv2Detector(
                    model_name=self.model_name,
                    objects=self.objects,
                    threshold=self.threshold
                )
            elif self.detector_type == 'owlvit':
                self.detector = OwlViTDetector(
                    model_name=self.model_name,
                    objects=self.objects,
                    threshold=self.threshold
                )
            else:
                raise ValueError("Invalid detector type")
        except Exception as e:
            print(f"Error initializing detector: {e}")
            self.detector = None
    
    def _initialize_video_capture(self):
        """Initialize video capture from specified device."""
        if self.cap:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(self.video_device_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video device {self.video_device_id}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

    def _typical_resolutions(self, device_id):
        common_res = [
            (1920, 1080), (1600, 900), (1280, 720), (1024, 576),
            (960, 540), (854, 480), (800, 600), (800, 450),
            (768, 576), (720, 480), (640, 480), (640, 360),
            (640, 240), (424, 240), (320, 240), (320, 180)
        ]

        return common_res
    
    def _render_detections_on_frame(self, frame, detections):
        """Draw bounding boxes and labels on the frame."""
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label_text = det["label"]
            confidence = det["score"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_text}: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
    
    def _capture_frames(self):
        """Continuously capture and process frames."""
        frame_count = 0
        
        try:
            self._initialize_video_capture()
        except RuntimeError as e:
            print(f"Failed to initialize video capture: {e}")
            self.running = False
            return
        
        detections = []

        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            if frame_count % self.frame_skip == 0 and self.detector:
                detections = self.detector.detect_objects(frame)
            
            frame_with_detections = self._render_detections_on_frame(frame.copy(), detections)
            
            with self.lock:
                self.current_frame = frame.copy()
                self.current_frame_with_detections = frame_with_detections.copy()
                self.last_detections = detections
            
            frame_count += 1
        
        self.cap.release()
    
    def start_capture(self):
        """Start the frame capture thread."""
        if not self.running:
            self.running = True
            self._capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self._capture_thread.start()
    
    def stop_capture(self):
        """Stop the frame capture thread."""
        if self.running:
            self.running = False
            self._capture_thread.join()
    
    def frame_to_jpeg(self, frame):
        """Convert OpenCV frame to JPEG bytes."""
        if frame is None:
            return None
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            return jpeg.tobytes()
        return None
    
    @cherrypy.expose
    def index(self):
        """Main page with links to all endpoints."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Object Detection Web App</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                nav { background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 30px; }
                nav a { display: inline-block; margin-right: 20px; text-decoration: none; color: #0066cc; }
                nav a:hover { text-decoration: underline; }
                .section { background-color: #f9f9f9; padding: 20px; margin: 20px 0; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Object Detection Web App</h1>
            <nav>
                <a href="/stream_with_detections">Stream with Detections</a>
                <a href="/stream_without_detections">Stream without Detections</a>
                <a href="/settings">Settings</a>
                <a href="/detections.json">Last Detections (JSON)</a>
            </nav>
            <div class="section">
                <h2>Available Pages</h2>
                <ul>
                    <li><a href="/stream_with_detections">Stream with Detections</a> - Webcam stream with bounding boxes</li>
                    <li><a href="/stream_without_detections">Stream without Detections</a> - Raw webcam stream</li>
                    <li><a href="/settings">Settings</a> - Configure detector parameters</li>
                    <li><a href="/detections.json">Last Detections (JSON)</a> - JSON with last detected objects</li>
                </ul>
            </div>
        </body>
        </html>
        """
    
    @cherrypy.expose
    def stream_with_detections(self):
        """Page showing webcam stream with detections."""
        if not self.running:
            self.start_capture()
        
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Stream with Detections</title>
        </head>
        <body>
            <img src="/video_feed_with_detections" alt="Stream with Detections">
        </body>
        </html>
        """
    
    @cherrypy.expose
    def stream_without_detections(self):
        """Page showing raw webcam stream."""
        if not self.running:
            self.start_capture()
        
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Stream without Detections</title>
            <style>
            </style>
        </head>
        <body>
            <img src="/video_feed_without_detections" alt="Stream without Detections">
        </body>
        </html>
        """
    
    @cherrypy.expose
    def video_feed_with_detections(self):
        """Stream video frames with detections as MJPEG."""
        cherrypy.response.headers['Content-Type'] = 'multipart/x-mixed-replace; boundary=frame'
        
        def generate():
            while True:
                with self.lock:
                    frame = self.current_frame_with_detections
                
                if frame is not None:
                    jpeg_data = self.frame_to_jpeg(frame)
                    if jpeg_data:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n'
                               b'Content-Length: ' + str(len(jpeg_data)).encode() + b'\r\n\r\n'
                               + jpeg_data + b'\r\n')
        
        return generate()
    
    video_feed_with_detections._cp_config = {'response.stream': True}
    
    @cherrypy.expose
    def video_feed_without_detections(self):
        """Stream video frames without detections as MJPEG."""
        cherrypy.response.headers['Content-Type'] = 'multipart/x-mixed-replace; boundary=frame'
        
        def generate():
            while True:
                with self.lock:
                    frame = self.current_frame
                
                if frame is not None:
                    jpeg_data = self.frame_to_jpeg(frame)
                    if jpeg_data:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n'
                               b'Content-Length: ' + str(len(jpeg_data)).encode() + b'\r\n\r\n'
                               + jpeg_data + b'\r\n')
        
        return generate()
    
    video_feed_without_detections._cp_config = {'response.stream': True}
    
    @cherrypy.expose
    def settings(self):
        """Settings page to configure detector parameters."""
        resolutions = self._typical_resolutions(self.video_device_id)
        resolution_options = []
        matched_current = False
        for w, h in resolutions:
            selected = ''
            if w == self.frame_width and h == self.frame_height:
                selected = 'selected'
                matched_current = True
            resolution_options.append(f"<option value=\"{w}x{h}\" {selected}>{w} x {h}</option>")
        custom_selected = '' if matched_current else 'selected'
        resolution_options.append(f"<option value=\"custom\" {custom_selected}>Custom</option>")
        resolution_options_html = "\n".join(resolution_options)

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Settings</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; max-width: 800px; }}
                h1 {{ color: #333; }}
                form {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; }}
                .form-group {{ margin-bottom: 15px; }}
                label {{ display: block; margin-bottom: 5px; font-weight: bold; }}
                input, select, textarea {{ width: 100%; padding: 8px; box-sizing: border-box; }}
                textarea {{ height: 100px; }}
                button {{ background-color: #0066cc; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }}
                button:hover {{ background-color: #0052a3; }}
                .info {{ background-color: #e7f3ff; padding: 10px; border-radius: 5px; margin: 20px 0; }}
                a {{ text-decoration: none; color: #0066cc; }}
                a:hover {{ text-decoration: underline; }}
                .inline {{ display: flex; gap: 10px; align-items: center; }}
                .inline input {{ width: 100%; }}
                .notification {{
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    padding: 15px 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                    z-index: 1000;
                    display: none;
                    min-width: 300px;
                    animation: slideIn 0.3s ease-out;
                }}
                .notification.success {{
                    background-color: #d4edda;
                    color: #155724;
                    border: 1px solid #c3e6cb;
                }}
                .notification.error {{
                    background-color: #f8d7da;
                    color: #721c24;
                    border: 1px solid #f5c6cb;
                }}
                @keyframes slideIn {{
                    from {{
                        transform: translateX(400px);
                        opacity: 0;
                    }}
                    to {{
                        transform: translateX(0);
                        opacity: 1;
                    }}
                }}
            </style>
            <script>
                function onResolutionChange(val) {{
                    if (val === 'custom') {{ return; }}
                    const parts = val.split('x');
                    if (parts.length === 2) {{
                        document.getElementById('frame_width').value = parts[0];
                        document.getElementById('frame_height').value = parts[1];
                    }}
                }}
                
                function showNotification(message, type) {{
                    const notification = document.getElementById('notification');
                    notification.textContent = message;
                    notification.className = 'notification ' + type;
                    notification.style.display = 'block';
                    
                    setTimeout(() => {{
                        notification.style.display = 'none';
                    }}, 5000);
                }}
                
                function handleSubmit(event) {{
                    event.preventDefault();
                    const form = event.target;
                    const formData = new FormData(form);
                    const submitButton = form.querySelector('button[type="submit"]');
                    
                    submitButton.disabled = true;
                    submitButton.textContent = 'Applying...';
                    
                    fetch('/apply_settings', {{
                        method: 'POST',
                        body: formData
                    }})
                    .then(response => response.json())
                    .then(data => {{
                        if (data.status === 'success') {{
                            showNotification(data.message, 'success');
                        }} else {{
                            showNotification(data.message, 'error');
                        }}
                    }})
                    .catch(error => {{
                        showNotification('Error: ' + error.message, 'error');
                    }})
                    .finally(() => {{
                        submitButton.disabled = false;
                        submitButton.textContent = 'Apply Settings';
                    }});
                }}
            </script>
        </head>
        <body>
            <div id="notification" class="notification"></div>
            <h1>Settings</h1>
            <div class="info">
                <p><strong>Note:</strong> Changes take effect after clicking "Apply Settings".</p>
            </div>
            <form method="POST" action="/apply_settings" onsubmit="handleSubmit(event)">
                <div class="form-group">
                    <label for="detector">Detector Type:</label>
                    <select name="detector" id="detector" required>
                        <option value="owlvit" {'selected' if self.detector_type == 'owlvit' else ''}>OWL-ViT</option>
                        <option value="owlv2" {'selected' if self.detector_type == 'owlv2' else ''}>OWL-v2</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="model_name">Model Name (leave blank for default):</label>
                    <input type="text" name="model_name" id="model_name" value="{self.model_name or ''}" />
                </div>
                <div class="form-group">
                    <label for="objects">Objects to Detect (one per line):</label>
                    <textarea name="objects" id="objects" required>{chr(10).join(self.objects)}</textarea>
                </div>
                <div class="form-group">
                    <label for="threshold">Detection Threshold (0.0 - 1.0):</label>
                    <input type="number" name="threshold" id="threshold" min="0" max="1" step="0.01" value="{self.threshold}" required />
                </div>
                <div class="form-group">
                    <label for="resolution_select">Predefined Resolutions:</label>
                    <select id="resolution_select" onchange="onResolutionChange(this.value)">
                        {resolution_options_html}
                    </select>
                </div>
                <div class="form-group inline">
                    <div style="flex:1;">
                        <label for="frame_width">Frame Width (pixels):</label>
                        <input type="number" name="frame_width" id="frame_width" min="1" value="{self.frame_width}" required />
                    </div>
                    <div style="flex:1;">
                        <label for="frame_height">Frame Height (pixels):</label>
                        <input type="number" name="frame_height" id="frame_height" min="1" value="{self.frame_height}" required />
                    </div>
                </div>
                <div class="form-group">
                    <label for="video_device_id">Video Device ID:</label>
                    <input type="number" name="video_device_id" id="video_device_id" min="0" value="{self.video_device_id}" required />
                </div>
                <div class="form-group">
                    <label for="frame_skip">Frame Skip (process every N frames):</label>
                    <input type="number" name="frame_skip" id="frame_skip" min="1" value="{self.frame_skip}" required />
                </div>
                <button type="submit">Apply Settings</button>
            </form>
            <br>
            <a href="/">← Back to Home</a>
        </body>
        </html>
        """
    
    @cherrypy.expose
    @cherrypy.tools.json_out()
    def apply_settings(self, **kwargs):
        """Apply new settings from the form."""
        try:
            # Extract and validate new settings
            new_detector_type = kwargs.get('detector', 'owlv2')
            new_model_name = kwargs.get('model_name') or None
            new_objects = [obj.strip() for obj in kwargs.get('objects', '').split('\n') if obj.strip()]
            new_threshold = float(kwargs.get('threshold', 0.17))
            new_frame_width = int(kwargs.get('frame_width', 640))
            new_frame_height = int(kwargs.get('frame_height', 480))
            new_video_device_id = int(kwargs.get('video_device_id', 0))
            new_frame_skip = int(kwargs.get('frame_skip', 5))
            
            # Check if model reload is needed (only detector type or model name changed)
            model_reload_needed = (
                new_detector_type != self.detector_type or
                new_model_name != self.model_name
            )
            
            # Check if detector parameters changed (but not objects/threshold)
            detector_params_changed = (
                new_objects != self.objects or
                new_threshold != self.threshold
            )
            
            # Check if video device changed
            video_device_changed = (
                new_video_device_id != self.video_device_id or
                new_frame_height != self.frame_height or
                new_frame_width != self.frame_width
            )
            
            # Apply all settings
            self.detector_type = new_detector_type
            self.model_name = new_model_name
            self.objects = new_objects
            self.threshold = new_threshold
            self.frame_width = new_frame_width
            self.frame_height = new_frame_height
            self.video_device_id = new_video_device_id
            self.frame_skip = new_frame_skip

            if video_device_changed:
                self.stop_capture()
                self.start_capture()
                        
            if model_reload_needed:
                # Need to reload the entire detector
                self._initialize_detector()
            elif detector_params_changed and self.detector:
                # Just update objects and threshold without reloading model
                self.detector.set_objects(new_objects)
                self.detector.set_threshold(new_threshold)
                        
            # Generate appropriate message based on what changed
            if model_reload_needed:
                model_reload_msg = "Settings applied successfully! The model has been reloaded."
            elif detector_params_changed:
                model_reload_msg = "Settings applied successfully! Detection parameters updated without reloading the model."
            else:
                model_reload_msg = "Settings updated successfully!"
            
            return {
                "status": "success",
                "message": model_reload_msg
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error applying settings: {str(e)}"
            }
    
    @cherrypy.expose
    @cherrypy.tools.json_out()
    def detections_json(self):
        """Return last detected objects as JSON."""
        with self.lock:
            detections = self.last_detections
        
        return {
            "status": "success",
            "model": { 
                "type":self.detector_type, 
                "name": self.model_name 
            },
            "frame": { 
                "width": self.frame_width,
                "height": self.frame_height
            },
            "threshold": self.threshold,
            "objects": self.objects,
            "detections": detections
        }

def app_exit(app):
    print("\nShutting down...")
    app.stop_capture()
    cherrypy.engine.exit()


def main():
    """Start the CherryPy web server."""
    app = ObjectDetectionApp()
    
    # Start frame capture
    app.start_capture()
    
    # Configure CherryPy
    config = {
        '/': {
            'tools.sessions.on': True,
            'tools.auth_basic.on': False,
        }
    }
    
    # Start server
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.tree.mount(app, '/', config)
    
    print("Starting Object Detection Web App on http://localhost:8080")
    print("Press Ctrl+C to stop the server")
    
    signal_handler = SignalHandler(cherrypy.engine)
    exit_fun = lambda: app_exit(app)
    signal_handler.handlers['SIGTERM'] = exit_fun
    signal_handler.handlers['SIGHUP'] = exit_fun
    signal_handler.handlers['SIGQUIT'] = exit_fun
    signal_handler.handlers['SIGINT'] = exit_fun
    signal_handler.subscribe()

    cherrypy.engine.start()
    cherrypy.engine.block()


if __name__ == '__main__':
    main()
