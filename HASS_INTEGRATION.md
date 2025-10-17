# Home Assistant MQTT Integration

This document describes how to integrate the TC001 thermal camera with Home Assistant using MQTT.

## Features

### 1. Camera Control
- **On/Off Switch**: Control camera operation remotely from Home Assistant
- **Automatic State Reporting**: Camera state (ON/OFF) is published to MQTT and synchronized with Home Assistant
- **On-Demand Operation**: Camera can be started and stopped dynamically, releasing resources when not in use

### 2. Temperature Monitoring
- **Max Temperature Sensor**: Publishes the maximum temperature detected by the camera every 60 seconds
- **Automatic Discovery**: All entities are automatically discovered by Home Assistant using MQTT discovery protocol
- **Device Grouping**: All entities are grouped under a single "TC001 Thermal Camera" device

### 3. MJPEG Streaming
- **Continuous Stream**: MJPEG stream remains available even when camera is toggled off (serves last frame)
- **Home Assistant Integration**: Can be integrated as a camera entity in Home Assistant
- **Port**: Configurable (default: 8080)

## Setup Instructions

### 1. Install Dependencies

**Option A: Using the provided virtual environment (recommended)**

The project includes a pre-configured virtual environment with all dependencies:

```bash
cd /home/scott/src/PyThermalCamera
source venv/bin/activate
```

**Option B: Manual installation**

If you need to recreate the virtual environment:

```bash
cd /home/scott/src/PyThermalCamera
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure MQTT Connection

Create a `config.yaml` file in the PyThermalCamera directory:

```yaml
# MQTT Configuration for Home Assistant Integration
mqtt_broker: "192.168.1.100"  # Your MQTT broker IP or hostname
mqtt_port: 1883
mqtt_username: "mqtt_user"  # Leave empty if no authentication
mqtt_password: "mqtt_password"  # Leave empty if no authentication
mqtt_client_id: "thermal_camera"

# Device information for Home Assistant
device_name: "TC001 Thermal Camera"
device_id: "thermal_camera"
manufacturer: "Topdon"
model: "TC001"
```

### 3. Run the Thermal Camera Application

**Option A: Using the helper script (easiest)**

```bash
# With MJPEG streaming and MQTT (headless)
./run.sh --stream --headless --config config.yaml

# Without streaming (MQTT only)
./run.sh --headless --config config.yaml

# With display window
./run.sh --stream --config config.yaml
```

**Option B: Direct Python execution**

```bash
# Activate virtual environment first
source venv/bin/activate

# Then run with desired options
python3 src/tc001v4.2.py --stream --headless --config config.yaml
```

### 4. Run as a System Service (Optional)

To run the thermal camera automatically on boot:

```bash
# Copy the service file to systemd
sudo cp thermal-camera.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable the service to start on boot
sudo systemctl enable thermal-camera.service

# Start the service now
sudo systemctl start thermal-camera.service

# Check status
sudo systemctl status thermal-camera.service
```

### 5. Home Assistant Configuration

#### Auto-Discovery (Recommended)

If you have MQTT discovery enabled in Home Assistant (default), the entities will appear automatically:
- Navigate to **Settings** → **Devices & Services** → **MQTT**
- Look for "TC001 Thermal Camera" device
- The device will have:
  - **Switch**: Camera (turn camera on/off)
  - **Sensor**: Max Temperature (temperature in °C)

#### Manual Camera Entity (for MJPEG stream)

Add to your `configuration.yaml`:

```yaml
camera:
  - platform: generic
    name: Thermal Camera
    still_image_url: http://192.168.1.XXX:8080/stream.mjpg
    stream_source: http://192.168.1.XXX:8080/stream.mjpg
```

Replace `192.168.1.XXX` with the IP address of the machine running the thermal camera application.

## MQTT Topics

### Published Topics (State)
- `thermal_camera/camera/state` - Camera state (ON/OFF)
- `thermal_camera/max_temp/state` - Maximum temperature (published every 60 seconds when camera is active)

### Subscribed Topics (Commands)
- `thermal_camera/camera/set` - Control camera (payload: ON or OFF)

### Discovery Topics
- `homeassistant/switch/thermal_camera/camera/config` - Switch discovery
- `homeassistant/sensor/thermal_camera/max_temp/config` - Sensor discovery

## Command Line Options

```
--device INT      Video device number (default: 0)
--stream          Enable MJPEG streaming server
--port INT        MJPEG streaming port (default: 8080)
--headless        Run without OpenCV window
--config PATH     Path to MQTT config file (default: config.yaml)
```

## Example Home Assistant Automations

### Turn camera on when motion detected
```yaml
automation:
  - alias: "Turn on thermal camera on motion"
    trigger:
      - platform: state
        entity_id: binary_sensor.motion_detector
        to: 'on'
    action:
      - service: switch.turn_on
        target:
          entity_id: switch.thermal_camera_camera
```

### Alert on high temperature
```yaml
automation:
  - alias: "Alert on high temperature"
    trigger:
      - platform: numeric_state
        entity_id: sensor.thermal_camera_max_temperature
        above: 50
    action:
      - service: notify.mobile_app
        data:
          message: "High temperature detected: {{ states('sensor.thermal_camera_max_temperature') }}°C"
```

## Troubleshooting

### MQTT not connecting
1. Verify MQTT broker is running: `mosquitto -v` (if using Mosquitto)
2. Check broker IP and port in `config.yaml`
3. Verify credentials if authentication is enabled
4. Check firewall rules for port 1883

### Entities not appearing in Home Assistant
1. Ensure MQTT integration is installed and configured in Home Assistant
2. Check Home Assistant logs: **Settings** → **System** → **Logs**
3. Verify MQTT discovery is enabled in Home Assistant configuration
4. Check MQTT topic structure using MQTT Explorer or similar tool

### Camera stream not working
1. Verify `--stream` flag is used when starting the application
2. Check the streaming port is not blocked by firewall
3. Try accessing stream directly: `http://CAMERA_IP:8080/stream.mjpg`
4. Ensure camera device is available: `v4l2-ctl --list-devices`

## Architecture

### CameraController Class
- Manages camera lifecycle (start/stop)
- Runs in separate thread for non-blocking operation
- Collects and stores temperature data with thread-safe access
- Handles image processing and display

### MQTTManager Class
- Manages MQTT connection and messaging
- Publishes Home Assistant discovery messages
- Handles camera control commands
- Publishes sensor data on 60-second interval

### StreamingServer
- Serves MJPEG stream over HTTP
- Thread-safe frame buffer
- Supports multiple concurrent clients
- Continues serving last frame when camera is stopped
