"""Publish 100 numbered samples at nominal 10 Hz; no hardware is required."""
import time
from pylsl import StreamInfo, StreamOutlet

outlet = StreamOutlet(StreamInfo("AIT-Demo", "Demo", 1, 10, "float32", "ait-bci-demo"))
print("Publishing AIT-Demo for 10 seconds", flush=True)
for index in range(100):
    outlet.push_sample([float(index)])
    time.sleep(0.1)
