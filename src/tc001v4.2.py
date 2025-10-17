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
		self.hud = True
		self.recording = False
		self.elapsed = "00:00:00"
		self.snaptime = "None"
		self.dispFullscreen = False

		# Camera state
		self.cap = None
		self.running = False
		self.thread = None

		# Latest sensor data
		self.maxtemp = None
		self.mintemp = None
		self.avgtemp = None
		self.centertemp = None
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
		print("Camera started")

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

	def get_latest_temps(self):
		with self.lock:
			return {
				'max': self.maxtemp,
				'min': self.mintemp,
				'avg': self.avgtemp,
				'center': self.centertemp
			}

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

			# Store temperatures
			with self.lock:
				self.maxtemp = maxtemp
				self.mintemp = mintemp
				self.avgtemp = avgtemp
				self.centertemp = centertemp

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
		self.lock = threading.Lock()

	def update_frame(self, frame):
		with self.lock:
			self.frame = frame.copy()

	def get_frame(self):
		with self.lock:
			if self.frame is None:
				return None
			return self.frame.copy()

class StreamingHandler(BaseHTTPRequestHandler):
	def do_GET(self):
		if self.path == '/stream.mjpg':
			self.send_response(200)
			self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
			self.send_header('Cache-Control', 'no-cache')
			self.send_header('Pragma', 'no-cache')
			self.end_headers()
			try:
				while True:
					frame = streaming_output.get_frame()
					if frame is not None:
						# Encode frame as JPEG
						ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
						if ret:
							self.wfile.write(b'--jpgboundary\r\n')
							self.send_header('Content-Type', 'image/jpeg')
							self.send_header('Content-Length', str(len(jpeg)))
							self.end_headers()
							self.wfile.write(jpeg.tobytes())
							self.wfile.write(b'\r\n')
					time.sleep(0.033)  # ~30 fps
			except Exception:
				pass
		elif self.path == '/':
			self.send_response(200)
			self.send_header('Content-Type', 'text/html')
			self.end_headers()
			html = b'''
			<html>
			<head><title>Thermal Camera Stream</title></head>
			<body>
			<h1>TC001 Thermal Camera Stream</h1>
			<img src="/stream.mjpg" />
			</body>
			</html>
			'''
			self.wfile.write(html)
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
		self.max_temp_state_topic = f"{self.base_topic}/max_temp/state"

		# Home Assistant discovery topics
		self.switch_config_topic = f"homeassistant/switch/{device_id}/camera/config"
		self.sensor_config_topic = f"homeassistant/sensor/{device_id}/max_temp/config"

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
					self._publish_camera_state()
				elif payload == "OFF":
					self.camera.stop()
					self._publish_camera_state()

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

		# Sensor discovery config
		sensor_config = {
			"name": "Max Temperature",
			"unique_id": f"{self.config.get('device_id', 'thermal_camera')}_max_temp",
			"state_topic": self.max_temp_state_topic,
			"unit_of_measurement": "°C",
			"device_class": "temperature",
			"state_class": "measurement",
			"device": device
		}

		# Publish discovery messages
		self.client.publish(self.switch_config_topic, json.dumps(switch_config), retain=True)
		self.client.publish(self.sensor_config_topic, json.dumps(sensor_config), retain=True)
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
			temps = self.camera.get_latest_temps()
			if temps['max'] is not None:
				self.client.publish(self.max_temp_state_topic, str(temps['max']))
				print(f"MQTT: Published max temp: {temps['max']}°C")
			self.last_publish_time = current_time

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
