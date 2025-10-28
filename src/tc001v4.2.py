#!/usr/bin/env python3
'''
Les Wright 21 June 2023
https://youtube.com/leslaboratory
A Python program to read, parse and display thermal data from the Topdon TC001 Thermal camera!
'''
print('Les Wright 21 June 2023')
print('https://youtube.com/leslaboratory')
print('A Python program to read, parse and display thermal data from the Topdon TC001 Thermal camera!')
print('')
print('Tested on Debian all features are working correctly')
print('This will work on the Pi However a number of workarounds are implemented!')
print('Seemingly there are bugs in the compiled version of cv2 that ships with the Pi!')
print('')
print('Key Bindings:')
print('')
print('a z: Increase/Decrease Blur')
print('s x: Floating High and Low Temp Label Threshold')
print('d c: Change Interpolated scale Note: This will not change the window size on the Pi')
print('f v: Contrast')
print('q w: Fullscreen Windowed (note going back to windowed does not seem to work on the Pi!)')
print('r t: Record and Stop')
print('p : Snapshot')
print('m : Cycle through ColorMaps')
print('h : Toggle HUD')

import cv2
import numpy as np
import argparse
import time
from datetime import datetime
import io
import threading
import socketserver
from http.server import BaseHTTPRequestHandler
import yaml
import paho.mqtt.client as mqtt
import json
import signal
import sys

#We need to know if we are running on the Pi, because openCV behaves a little oddly on all the builds!
#https://raspberrypi.stackexchange.com/questions/5100/detect-that-a-python-program-is-running-on-the-pi
def is_raspberrypi():
    try:
        with io.open('/sys/firmware/devicetree/base/model', 'r') as m:
            if 'raspberry pi' in m.read().lower(): return True
    except Exception: pass
    return False

isPi = is_raspberrypi()

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="Video Device number e.g. 0, use v4l2-ctl --list-devices")
parser.add_argument("--stream", action="store_true", help="Enable MJPEG streaming server")
parser.add_argument("--port", type=int, default=8080, help="Port for MJPEG streaming server (default: 8080)")
parser.add_argument("--headless", action="store_true", help="Run without OpenCV window (headless mode)")
parser.add_argument("--config", type=str, default="config.yaml", help="Path to MQTT configuration file (default: config.yaml)")
args = parser.parse_args()

# Load MQTT configuration
mqtt_config = {}
try:
	with open(args.config, 'r') as f:
		mqtt_config = yaml.safe_load(f)
		print(f'Loaded MQTT config from {args.config}')
except FileNotFoundError:
	print(f'Config file {args.config} not found. MQTT features will be disabled.')
	mqtt_config = None
except Exception as e:
	print(f'Error loading config file: {e}. MQTT features will be disabled.')
	mqtt_config = None
	
# Get device number from args
if args.device:
	dev = args.device
else:
	dev = 0

# Global settings (kept for backward compatibility with rec() and snapshot() functions)
width = 256
height = 192
scale = 3
newWidth = width*scale
newHeight = height*scale

def rec():
	now = time.strftime("%Y%m%d--%H%M%S")
	#do NOT use mp4 here, it is flakey!
	videoOut = cv2.VideoWriter(now+'output.avi', cv2.VideoWriter_fourcc(*'XVID'),25, (newWidth,newHeight))
	return(videoOut)

def snapshot(heatmap):
	#I would put colons in here, but it Win throws a fit if you try and open them!
	now = time.strftime("%Y%m%d-%H%M%S")
	snaptime = time.strftime("%H:%M:%S")
	cv2.imwrite("TC001"+now+".png", heatmap)
	return snaptime

def celsius_to_fahrenheit(celsius):
	"""Convert Celsius to Fahrenheit"""
	return (celsius * 9/5) + 32

# Camera Controller Class
class CameraController:
	def __init__(self, device_num, headless=False, width=256, height=192, scale=3):
		self.device_num = device_num
		self.headless = headless
		self.width = width
		self.height = height
		self.scale = scale
		self.newWidth = width * scale
		self.newHeight = height * scale
		self.alpha = 1.0
		self.colormap = 0
		self.rad = 0
		self.threshold = 2
		self.hud = False
		self.recording = False
		self.elapsed = "00:00:00"
		self.snaptime = "None"
		self.dispFullscreen = False

		# Camera state
		self.cap = None
		self.running = False
		self.thread = None
		self.mqtt_manager = None  # Will be set externally
		self.start_time = time.time()

		# Latest sensor data
		self.temps = {
			'max': None,
			'min': None,
			'avg': None,
			'center': None,
			'p50': None,
			'p90': None,
		}
		self.lock = threading.Lock()

	def start(self):
		if self.running:
			return

		# Initialize video capture
		self.cap = cv2.VideoCapture(f'/dev/video{self.device_num}', cv2.CAP_V4L)
		if isPi:
			self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
		else:
			self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)

		# Create window if not headless
		if not self.headless and not args.headless:
			cv2.namedWindow('Thermal', cv2.WINDOW_GUI_NORMAL)
			cv2.resizeWindow('Thermal', self.newWidth, self.newHeight)

		self.running = True
		self.thread = threading.Thread(target=self._capture_loop, daemon=True)
		self.thread.start()
		self.start_time = time.time()
		print("Camera started")

		# Publish state change to MQTT
		if self.mqtt_manager:
			self.mqtt_manager._publish_camera_state()

		# Broadcast state change to SSE clients
		broadcast_camera_state("on")

	def stop(self):
		if not self.running:
			return

		self.running = False
		if self.thread:
			self.thread.join(timeout=2.0)

		if self.cap:
			self.cap.release()
			self.cap = None

		if not self.headless and not args.headless:
			cv2.destroyAllWindows()

		print("Camera stopped")

		# Create and publish "Camera Disabled" frame
		if args.stream and streaming_output is not None:
			disabled_frame = np.zeros((self.newHeight, self.newWidth, 3), dtype=np.uint8)
			cv2.putText(disabled_frame, 'CAMERA DISABLED',
					   (int(self.newWidth/2) - 150, int(self.newHeight/2)),
					   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
			streaming_output.update_frame(disabled_frame)

		# Publish state change to MQTT
		if self.mqtt_manager:
			self.mqtt_manager._publish_camera_state()

		# Broadcast state change to SSE clients
		broadcast_camera_state("off")

	def get_latest_temps(self):
		if not self.running:
			return None
		with self.lock:
			result = self.temps.copy()
			result['time_since_start'] = time.time() - self.start_time
			return result

	def _capture_loop(self):
		while self.running and self.cap and self.cap.isOpened():
			ret, frame = self.cap.read()
			if not ret:
				time.sleep(0.01)
				continue

			imdata, thdata = np.array_split(frame, 2)

			# Calculate center temperature
			hi = thdata[96][128][0]
			lo = thdata[96][128][1]
			lo = float(lo) * 256
			rawtemp = float(hi) + lo
			centertemp = (rawtemp / 64) - 273.15
			centertemp = round(centertemp, 2)

			# Calculate temperature for all pixels
			# Combine hi and lo bytes for raw temperature values
			raw_temp_array = (thdata[..., 0].astype(float) + (thdata[..., 1].astype(float) * 256))
			# Convert to Celsius
			temp_array = (raw_temp_array / 64) - 273.15

			# Find max temperature
			maxtemp = float(temp_array.max())
			posmax = temp_array.argmax()
			mcol, mrow = divmod(posmax, self.width)
			maxtemp = round(maxtemp, 2)

			# Find min temperature
			mintemp = float(temp_array.min())
			posmin = temp_array.argmin()
			lcol, lrow = divmod(posmin, self.width)
			mintemp = round(mintemp, 2)

			# Find average temperature (use the already calculated temp_array)
			avgtemp = float(temp_array.mean())
			avgtemp = round(avgtemp, 2)

			# Calculate percentiles efficiently (all at once)
			percentiles = np.percentile(temp_array, [50, 90])
			p50temp = round(float(percentiles[0]), 2)
			p90temp = round(float(percentiles[1]), 2)

			# Store temperatures
			with self.lock:
				self.temps['max'] = maxtemp
				self.temps['min'] = mintemp
				self.temps['avg'] = avgtemp
				self.temps['center'] = centertemp
				self.temps['p50'] = p50temp
				self.temps['p90'] = p90temp

			# Convert image to RGB and apply processing
			bgr = cv2.cvtColor(imdata, cv2.COLOR_YUV2BGR_YUYV)
			bgr = cv2.convertScaleAbs(bgr, alpha=self.alpha)
			bgr = cv2.resize(bgr, (self.newWidth, self.newHeight), interpolation=cv2.INTER_CUBIC)
			if self.rad > 0:
				bgr = cv2.blur(bgr, (self.rad, self.rad))

			# Apply colormap
			colormap_list = [
				(cv2.COLORMAP_JET, 'Jet'),
				(cv2.COLORMAP_HOT, 'Hot'),
				(cv2.COLORMAP_MAGMA, 'Magma'),
				(cv2.COLORMAP_INFERNO, 'Inferno'),
				(cv2.COLORMAP_PLASMA, 'Plasma'),
				(cv2.COLORMAP_BONE, 'Bone'),
				(cv2.COLORMAP_SPRING, 'Spring'),
				(cv2.COLORMAP_AUTUMN, 'Autumn'),
				(cv2.COLORMAP_VIRIDIS, 'Viridis'),
				(cv2.COLORMAP_PARULA, 'Parula'),
				(cv2.COLORMAP_RAINBOW, 'Inv Rainbow')
			]
			cmap, cmapText = colormap_list[self.colormap % len(colormap_list)]
			heatmap = cv2.applyColorMap(bgr, cmap)
			if self.colormap == 10:
				heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

			# Draw crosshairs and center temp
			cv2.line(heatmap, (int(self.newWidth/2), int(self.newHeight/2)+20),
					(int(self.newWidth/2), int(self.newHeight/2)-20), (255, 255, 255), 2)
			cv2.line(heatmap, (int(self.newWidth/2)+20, int(self.newHeight/2)),
					(int(self.newWidth/2)-20, int(self.newHeight/2)), (255, 255, 255), 2)
			cv2.line(heatmap, (int(self.newWidth/2), int(self.newHeight/2)+20),
					(int(self.newWidth/2), int(self.newHeight/2)-20), (0, 0, 0), 1)
			cv2.line(heatmap, (int(self.newWidth/2)+20, int(self.newHeight/2)),
					(int(self.newWidth/2)-20, int(self.newHeight/2)), (0, 0, 0), 1)
			# Convert to Fahrenheit for display
			centertemp_f = round(celsius_to_fahrenheit(centertemp), 1)
			cv2.putText(heatmap, str(centertemp_f)+' F', (int(self.newWidth/2)+10, int(self.newHeight/2)-10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
			cv2.putText(heatmap, str(centertemp_f)+' F', (int(self.newWidth/2)+10, int(self.newHeight/2)-10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

			# Draw HUD if enabled
			if self.hud:
				cv2.rectangle(heatmap, (0, 0), (160, 120), (0, 0, 0), -1)
				# Convert temps to Fahrenheit for display
				avgtemp_f = round(celsius_to_fahrenheit(avgtemp), 1)
				threshold_f = round(celsius_to_fahrenheit(self.threshold), 1)
				cv2.putText(heatmap, 'Avg Temp: '+str(avgtemp_f)+' F', (10, 14),
						cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
				cv2.putText(heatmap, 'Label Threshold: '+str(threshold_f)+' F', (10, 28),
						cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
				cv2.putText(heatmap, 'Colormap: '+cmapText, (10, 42),
						cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
				cv2.putText(heatmap, 'Blur: '+str(self.rad)+' ', (10, 56),
						cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
				cv2.putText(heatmap, 'Scaling: '+str(self.scale)+' ', (10, 70),
						cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
				cv2.putText(heatmap, 'Contrast: '+str(self.alpha)+' ', (10, 84),
						cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
				cv2.putText(heatmap, 'Snapshot: '+self.snaptime+' ', (10, 98),
						cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

			# Display floating max/min temps (convert to Fahrenheit)
			if maxtemp > avgtemp + self.threshold:
				maxtemp_f = round(celsius_to_fahrenheit(maxtemp), 1)
				cv2.circle(heatmap, (mrow*self.scale, mcol*self.scale), 5, (0, 0, 0), 2)
				cv2.circle(heatmap, (mrow*self.scale, mcol*self.scale), 5, (0, 0, 255), -1)
				cv2.putText(heatmap, str(maxtemp_f)+' F', ((mrow*self.scale)+10, (mcol*self.scale)+5),
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
				cv2.putText(heatmap, str(maxtemp_f)+' F', ((mrow*self.scale)+10, (mcol*self.scale)+5),
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

			if mintemp < avgtemp - self.threshold:
				mintemp_f = round(celsius_to_fahrenheit(mintemp), 1)
				cv2.circle(heatmap, (lrow*self.scale, lcol*self.scale), 5, (0, 0, 0), 2)
				cv2.circle(heatmap, (lrow*self.scale, lcol*self.scale), 5, (255, 0, 0), -1)
				cv2.putText(heatmap, str(mintemp_f)+' F', ((lrow*self.scale)+10, (lcol*self.scale)+5),
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
				cv2.putText(heatmap, str(mintemp_f)+' F', ((lrow*self.scale)+10, (lcol*self.scale)+5),
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

			# Add timestamp overlay in bottom left corner
			timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]  # ISO8601 with milliseconds
			text_y = self.newHeight - 10  # 10 pixels from bottom
			cv2.putText(heatmap, timestamp, (10, text_y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
			cv2.putText(heatmap, timestamp, (10, text_y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

			# Display image if not headless
			if not self.headless and not args.headless:
				cv2.imshow('Thermal', heatmap)
				cv2.waitKey(1)

			# Update streaming buffer if enabled
			if args.stream and streaming_output is not None:
				streaming_output.update_frame(heatmap)

			time.sleep(0.01)

# MJPEG Streaming Infrastructure
class StreamingOutput:
	def __init__(self):
		self.frame = None
		self.frame_id = 0
		self.lock = threading.Lock()
		self.frame_event = threading.Event()

	def update_frame(self, frame):
		with self.lock:
			self.frame = frame.copy()
			self.frame_id += 1
		self.frame_event.set()

	def get_frame_with_id(self):
		"""Get current frame along with its ID"""
		with self.lock:
			if self.frame is None:
				return None, 0
			return self.frame.copy(), self.frame_id

	def wait_for_new_frame(self, timeout=1.0):
		"""Wait for a new frame to be available"""
		self.frame_event.clear()
		return self.frame_event.wait(timeout)

# SSE client management for real-time state updates
sse_clients = []
sse_clients_lock = threading.Lock()

def broadcast_camera_state(state):
	"""Broadcast camera state to all connected SSE clients"""
	with sse_clients_lock:
		disconnected = []
		for client in sse_clients:
			try:
				msg = f"data: {state}\n\n"
				client.wfile.write(msg.encode())
				client.wfile.flush()
			except Exception:
				disconnected.append(client)
		# Remove disconnected clients
		for client in disconnected:
			sse_clients.remove(client)

class StreamingHandler(BaseHTTPRequestHandler):
	def do_GET(self):
		if self.path == '/stream.mjpg':
			self.send_response(200)
			self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
			self.send_header('Cache-Control', 'no-cache')
			self.send_header('Pragma', 'no-cache')
			self.end_headers()
			try:
				last_frame_id = 0
				while True:
					# Wait for a new frame to be available
					streaming_output.wait_for_new_frame(timeout=1.0)

					# Get the frame with its ID
					frame, frame_id = streaming_output.get_frame_with_id()

					# Only send if we have a new frame
					if frame is not None and frame_id > last_frame_id:
						# Encode frame as JPEG
						ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
						if ret:
							# Write the JPEG data with proper multipart boundaries
							self.wfile.write(b'Content-Type: image/jpeg\r\n')
							self.wfile.write(f'Content-Length: {len(jpeg)}\r\n\r\n'.encode())
							self.wfile.write(jpeg.tobytes())
							self.wfile.write(b'\r\n--jpgboundary\r\n')
							last_frame_id = frame_id
			except Exception:
				pass
		elif self.path == '/events':
			# SSE endpoint for real-time camera state updates
			self.send_response(200)
			self.send_header('Content-Type', 'text/event-stream')
			self.send_header('Cache-Control', 'no-cache')
			self.send_header('Connection', 'keep-alive')
			self.end_headers()

			# Add this client to SSE clients list
			with sse_clients_lock:
				sse_clients.append(self)

			# Send initial state
			try:
				state = "on" if camera.running else "off"
				msg = f"data: {state}\n\n"
				self.wfile.write(msg.encode())
				self.wfile.flush()

				# Keep connection alive
				while True:
					time.sleep(30)
					# Send keep-alive comment
					self.wfile.write(b": keep-alive\n\n")
					self.wfile.flush()
			except Exception:
				pass
			finally:
				# Remove client when disconnected
				with sse_clients_lock:
					if self in sse_clients:
						sse_clients.remove(self)
		elif self.path == '/':
			self.send_response(200)
			self.send_header('Content-Type', 'text/html')
			self.end_headers()
			html = b'''
			<html>
			<head>
				<title>Thermal Camera Stream</title>
				<style>
					button:disabled { opacity: 0.5; }
				</style>
			</head>
			<body>
				<h1>TC001 Thermal Camera Stream</h1>
				<p>Status: <span id="status">...</span></p>
				<button id="onBtn">Turn ON</button>
				<button id="offBtn">Turn OFF</button>
				<br><br>
				<img src="/stream.mjpg" />
				<script>
					const status = document.getElementById('status');
					const onBtn = document.getElementById('onBtn');
					const offBtn = document.getElementById('offBtn');

					function updateUI(state) {
						status.textContent = state === 'on' ? 'ON' : 'OFF';
						onBtn.disabled = state === 'on';
						offBtn.disabled = state === 'off';
					}

					async function sendControl(action) {
						try {
							const response = await fetch('/control', {
								method: 'POST',
								headers: { 'Content-Type': 'application/json' },
								body: JSON.stringify({ action: action })
							});
							const data = await response.json();
							if (data.status === 'ok') {
								updateUI(data.state);
							}
						} catch (e) {
							console.error('Control error:', e);
						}
					}

					onBtn.addEventListener('click', () => sendControl('start'));
					offBtn.addEventListener('click', () => sendControl('stop'));

					// Connect to SSE for real-time updates
					const events = new EventSource('/events');
					events.onmessage = (e) => updateUI(e.data);
					events.onerror = () => console.error('SSE connection error');
				</script>
			</body>
			</html>
			'''
			self.wfile.write(html)
		else:
			self.send_response(404)
			self.end_headers()

	def do_POST(self):
		if self.path == '/control':
			# Read POST data
			content_length = int(self.headers.get('Content-Length', 0))
			post_data = self.rfile.read(content_length)

			try:
				# Parse JSON
				data = json.loads(post_data.decode())
				action = data.get('action')

				if action == 'start':
					camera.start()
					self.send_response(200)
					self.send_header('Content-Type', 'application/json')
					self.end_headers()
					self.wfile.write(json.dumps({'status': 'ok', 'state': 'on'}).encode())
				elif action == 'stop':
					camera.stop()
					self.send_response(200)
					self.send_header('Content-Type', 'application/json')
					self.end_headers()
					self.wfile.write(json.dumps({'status': 'ok', 'state': 'off'}).encode())
				else:
					self.send_response(400)
					self.send_header('Content-Type', 'application/json')
					self.end_headers()
					self.wfile.write(json.dumps({'status': 'error', 'message': 'Invalid action'}).encode())
			except Exception as e:
				self.send_response(500)
				self.send_header('Content-Type', 'application/json')
				self.end_headers()
				self.wfile.write(json.dumps({'status': 'error', 'message': str(e)}).encode())
		else:
			self.send_response(404)
			self.end_headers()

	def log_message(self, format, *args):
		# Suppress HTTP server logs
		pass

class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
	allow_reuse_address = True
	daemon_threads = True

# MQTT Manager for Home Assistant Integration
class MQTTManager:
	def __init__(self, config, camera_controller):
		self.config = config
		self.camera = camera_controller
		self.client = None
		self.connected = False
		self.last_publish_time = 0
		self.publish_interval = 60  # Publish every 60 seconds

		# MQTT topics
		device_id = config.get('device_id', 'thermal_camera')
		self.base_topic = f"thermal_camera"
		self.camera_command_topic = f"{self.base_topic}/camera/set"
		self.camera_state_topic = f"{self.base_topic}/camera/state"

		# Temperature sensor metadata (key: sensor_key, value: human-readable name)
		self.temp_sensors = {
			'max': 'Max Temperature',
			'min': 'Min Temperature',
			'p50': 'P50 Temperature (Median)',
			'p90': 'P90 Temperature',
		}

		# Generate state and config topics for all temperature sensors
		self.temp_state_topics = {}
		self.temp_config_topics = {}
		for sensor_key in self.temp_sensors.keys():
			self.temp_state_topics[sensor_key] = f"{self.base_topic}/{sensor_key}_temp/state"
			self.temp_config_topics[sensor_key] = f"homeassistant/sensor/{device_id}/{sensor_key}_temp/config"

		# Home Assistant discovery topics
		self.switch_config_topic = f"homeassistant/switch/{device_id}/camera/config"

	def connect(self):
		try:
			self.client = mqtt.Client(client_id=self.config.get('mqtt_client_id', 'thermal_camera'))

			# Set authentication if provided
			username = self.config.get('mqtt_username', '')
			password = self.config.get('mqtt_password', '')
			if username:
				self.client.username_pw_set(username, password)

			# Set up callbacks
			self.client.on_connect = self._on_connect
			self.client.on_message = self._on_message
			self.client.on_disconnect = self._on_disconnect

			# Connect to broker
			broker = self.config.get('mqtt_broker', 'localhost')
			port = self.config.get('mqtt_port', 1883)
			self.client.connect(broker, port, 60)

			# Start network loop in background thread
			self.client.loop_start()
			print(f"MQTT: Connecting to {broker}:{port}")

		except Exception as e:
			print(f"MQTT: Failed to connect: {e}")
			self.client = None

	def _on_connect(self, client, userdata, flags, rc):
		if rc == 0:
			self.connected = True
			print("MQTT: Connected successfully")

			# Subscribe to camera command topic
			self.client.subscribe(self.camera_command_topic)
			print(f"MQTT: Subscribed to {self.camera_command_topic}")

			# Publish Home Assistant discovery messages
			self._publish_discovery()

			# Publish initial state
			self._publish_camera_state()

		else:
			print(f"MQTT: Connection failed with code {rc}")

	def _on_disconnect(self, client, userdata, rc):
		self.connected = False
		print("MQTT: Disconnected")

	def _on_message(self, client, userdata, msg):
		try:
			payload = msg.payload.decode('utf-8')
			print(f"MQTT: Received message on {msg.topic}: {payload}")

			if msg.topic == self.camera_command_topic:
				if payload == "ON":
					self.camera.start()
					# State will be published by camera.start()
				elif payload == "OFF":
					self.camera.stop()
					# State will be published by camera.stop()

		except Exception as e:
			print(f"MQTT: Error processing message: {e}")

	def _publish_discovery(self):
		# Device information
		device = {
			"identifiers": [self.config.get('device_id', 'thermal_camera')],
			"name": self.config.get('device_name', 'TC001 Thermal Camera'),
			"manufacturer": self.config.get('manufacturer', 'Topdon'),
			"model": self.config.get('model', 'TC001')
		}

		# Switch discovery config
		switch_config = {
			"name": "Camera",
			"unique_id": f"{self.config.get('device_id', 'thermal_camera')}_camera",
			"command_topic": self.camera_command_topic,
			"state_topic": self.camera_state_topic,
			"payload_on": "ON",
			"payload_off": "OFF",
			"state_on": "ON",
			"state_off": "OFF",
			"device": device
		}

		# Publish discovery messages
		self.client.publish(self.switch_config_topic, json.dumps(switch_config), retain=True)

		# Publish temperature sensor discovery configs
		for sensor_key, sensor_name in self.temp_sensors.items():
			sensor_config = {
				"name": sensor_name,
				"unique_id": f"{self.config.get('device_id', 'thermal_camera')}_{sensor_key}_temp",
				"state_topic": self.temp_state_topics[sensor_key],
				"unit_of_measurement": "°C",
				"device_class": "temperature",
				"state_class": "measurement",
				"device": device
			}
			self.client.publish(self.temp_config_topics[sensor_key], json.dumps(sensor_config), retain=True)

		print("MQTT: Published Home Assistant discovery messages")

	def _publish_camera_state(self):
		if not self.connected:
			return

		state = "ON" if self.camera.running else "OFF"
		self.client.publish(self.camera_state_topic, state, retain=True)
		print(f"MQTT: Published camera state: {state}")

	def update(self):
		if not self.connected or not self.camera.running:
			return

		# Publish sensor data every 60 seconds
		current_time = time.time()
		if current_time - self.last_publish_time >= self.publish_interval:
			self.last_publish_time = current_time
			temps = self.camera.get_latest_temps()
			if temps is not None and temps['time_since_start'] > 60:
				# Publish all temperature sensors
				for sensor_key in self.temp_sensors.keys():
					if temps.get(sensor_key) is not None:
						self.client.publish(self.temp_state_topics[sensor_key], str(temps[sensor_key]))
						temp_f = round(celsius_to_fahrenheit(temps[sensor_key]), 1)
						print(f"MQTT: Published temperature stat - {sensor_key}: {temps[sensor_key]}°C ({temp_f}°F)")

	def disconnect(self):
		if self.client:
			self.client.loop_stop()
			self.client.disconnect()
			print("MQTT: Disconnected")

# Initialize streaming output if streaming is enabled
streaming_output = None
if args.stream:
	streaming_output = StreamingOutput()
	server = ThreadedHTTPServer(('0.0.0.0', args.port), StreamingHandler)
	server_thread = threading.Thread(target=server.serve_forever, daemon=True)
	server_thread.start()
	print(f'MJPEG streaming server started on port {args.port}')
	print(f'Access stream at: http://localhost:{args.port}/stream.mjpg')
	print(f'Or view in browser at: http://localhost:{args.port}/')

# Initialize camera controller
camera = CameraController(dev, args.headless)

# Initialize MQTT manager if config is available
mqtt_manager = None
if mqtt_config:
	mqtt_manager = MQTTManager(mqtt_config, camera)
	camera.mqtt_manager = mqtt_manager  # Give camera reference to mqtt_manager
	mqtt_manager.connect()
	# Start camera automatically if MQTT is enabled
	camera.start()
else:
	# Start camera immediately if no MQTT
	camera.start()

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
	print("\nShutting down gracefully...")
	if camera:
		camera.stop()
	if mqtt_manager:
		mqtt_manager.disconnect()
	sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Main loop - just keep the program running and update MQTT
print("Thermal camera system running. Press Ctrl+C to exit.")
try:
	while True:
		if mqtt_manager:
			mqtt_manager.update()
		time.sleep(1)
except KeyboardInterrupt:
	pass
finally:
	if camera:
		camera.stop()
	if mqtt_manager:
		mqtt_manager.disconnect()
	print("Exiting...")
