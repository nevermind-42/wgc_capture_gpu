import sys
import wgc_capture
import time
import numpy as np
import matplotlib.pyplot as plt
import threading
import cv2

# Don't show frames by default
DISPLAY_FRAMES = '--show' in sys.argv
TEST_DURATION = 10  # seconds
FPS_WINDOW = 1.0  # seconds for sliding FPS window

def wait_first_frame(cap, method='get_frame', timeout=5):
    print(f"Waiting for first frame ({method})...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        frame = cap.get_frame()
        if frame is not None and hasattr(frame, 'shape') and frame.size > 0:
            print("First frame received!")
            return frame
        time.sleep(0.01)
    print("Failed to get first frame within timeout!")
    return None

def test_get_frame():
    print("\nTesting get_frame() (main thread)...")
    cap = wgc_capture.WGCCapture()
    first = wait_first_frame(cap, 'get_frame')
    if first is None:
        return None
    timestamps = []
    delays = []
    frame_count = 0
    non_empty_frame_count = 0
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < TEST_DURATION:
        frame = cap.get_frame()
        now = time.time()
        if frame is not None and hasattr(frame, 'shape') and frame.size > 0:
            frame_count += 1
            if not np.all(frame == 0):  # Check for non-empty frame
                non_empty_frame_count += 1
            timestamps.append(now)
            delays.append(now - last_time)
            last_time = now
            if DISPLAY_FRAMES:
                if frame.shape[2] == 4:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    cv2.imshow('get_frame', frame_bgr)
                else:
                    cv2.imshow('get_frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    cv2.destroyAllWindows()
    print(f"get_frame: {frame_count} frames in {TEST_DURATION} sec, average FPS: {frame_count/TEST_DURATION:.2f}")
    print(f"get_frame: {non_empty_frame_count} non-empty frames, non-empty FPS: {non_empty_frame_count/TEST_DURATION:.2f}")
    return {'method': 'get_frame', 'timestamps': timestamps, 'delays': delays, 'total_frames': frame_count, 'non_empty_frames': non_empty_frame_count}

def test_polling_callback():
    print("\nTesting polling_callback (get_frame in separate thread)...")
    cap = wgc_capture.WGCCapture()
    first = wait_first_frame(cap, 'polling_callback')
    if first is None:
        return None
    timestamps = []
    delays = []
    frame_count = 0
    non_empty_frame_count = 0
    start_time = time.time()
    last_time = start_time
    stop_event = threading.Event()
    def poller():
        nonlocal frame_count, last_time, non_empty_frame_count
        while not stop_event.is_set() and time.time() - start_time < TEST_DURATION:
            frame = cap.get_frame()
            now = time.time()
            if frame is not None and hasattr(frame, 'shape') and frame.size > 0:
                frame_count += 1
                if not np.all(frame == 0):  # Check for non-empty frame
                    non_empty_frame_count += 1
                timestamps.append(now)
                delays.append(now - last_time)
                last_time = now
                if DISPLAY_FRAMES:
                    if frame.shape[2] == 4:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                        cv2.imshow('polling_callback', frame_bgr)
                    else:
                        cv2.imshow('polling_callback', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_event.set()
            time.sleep(0.001)
    t = threading.Thread(target=poller)
    t.start()
    t.join()
    cv2.destroyAllWindows()
    print(f"polling_callback: {frame_count} frames in {TEST_DURATION} sec, average FPS: {frame_count/TEST_DURATION:.2f}")
    print(f"polling_callback: {non_empty_frame_count} non-empty frames, non-empty FPS: {non_empty_frame_count/TEST_DURATION:.2f}")
    return {'method': 'polling_callback', 'timestamps': timestamps, 'delays': delays, 'total_frames': frame_count, 'non_empty_frames': non_empty_frame_count}

def plot_results(get_frame_results, polling_callback_results):
    plt.figure(figsize=(16, 12))
    # FPS over time (sliding window)
    plt.subplot(3,2,1)
    for res, color in zip([get_frame_results, polling_callback_results], ['b', 'r']):
        if res is None or not res['timestamps']:
            continue
        times = np.array(res['timestamps']) - res['timestamps'][0]
        fps = []
        t_grid = np.arange(0, times[-1], 0.1)
        for t in t_grid:
            fps.append(np.sum((times >= t) & (times < t+FPS_WINDOW))/FPS_WINDOW)
        plt.plot(t_grid, fps, color+'-', label=res['method'])
    plt.title('FPS over time (1s sliding window)')
    plt.xlabel('Time, sec')
    plt.ylabel('FPS')
    plt.legend()
    plt.grid(True)
    
    # Frame delay histogram
    plt.subplot(3,2,2)
    for res, color in zip([get_frame_results, polling_callback_results], ['b', 'r']):
        if res is None or not res['delays']:
            continue
        plt.hist(np.array(res['delays'])[1:]*1000, bins=50, alpha=0.5, color=color, label=res['method'])
    plt.title('Frame delay distribution')
    plt.xlabel('Delay, ms')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    # Average FPS histogram
    plt.subplot(3,2,3)
    methods = []
    avg_fps = []
    for res in [get_frame_results, polling_callback_results]:
        if res is not None:
            methods.append(res['method'])
            avg_fps.append(res['total_frames']/TEST_DURATION)
    plt.bar(methods, avg_fps, color=['blue','red'])
    plt.title('Average FPS (all frames)')
    plt.ylabel('FPS')
    for i, v in enumerate(avg_fps):
        plt.text(i, v+0.5, f"{v:.1f}", ha='center')
    plt.grid(True, axis='y')
    
    # Non-empty frame FPS histogram
    plt.subplot(3,2,4)
    non_empty_fps = []
    for res in [get_frame_results, polling_callback_results]:
        if res is not None:
            non_empty_fps.append(res['non_empty_frames']/TEST_DURATION)
    plt.bar(methods, non_empty_fps, color=['blue','red'])
    plt.title('Average FPS (non-empty frames only)')
    plt.ylabel('FPS')
    for i, v in enumerate(non_empty_fps):
        plt.text(i, v+0.5, f"{v:.1f}", ha='center')
    plt.grid(True, axis='y')
    
    # Average delay histogram
    plt.subplot(3,2,5)
    avg_delay = []
    for res in [get_frame_results, polling_callback_results]:
        if res is not None and len(res['delays']) > 1:
            avg_delay.append(np.mean(res['delays'][1:])*1000)
        else:
            avg_delay.append(0)
    plt.bar(methods, avg_delay, color=['blue','red'])
    plt.title('Average frame delay')
    plt.ylabel('Delay, ms')
    for i, v in enumerate(avg_delay):
        plt.text(i, v+0.5, f"{v:.1f}", ha='center')
    plt.grid(True, axis='y')
    
    # Non-empty frame percentage
    plt.subplot(3,2,6)
    non_empty_percent = []
    for res in [get_frame_results, polling_callback_results]:
        if res is not None:
            non_empty_percent.append(res['non_empty_frames']/res['total_frames']*100)
    plt.bar(methods, non_empty_percent, color=['blue','red'])
    plt.title('Non-empty frame percentage')
    plt.ylabel('Percentage')
    for i, v in enumerate(non_empty_percent):
        plt.text(i, v+0.5, f"{v:.1f}%", ha='center')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('capture_performance_comparison.png')
    plt.show()

def print_summary(res):
    if res is None:
        print("No data available for this method!")
        return
    print(f"\nMethod: {res['method']}")
    print(f"  Frames: {res['total_frames']} in {TEST_DURATION} sec")
    print(f"  Average FPS: {res['total_frames']/TEST_DURATION:.2f}")
    print(f"  Non-empty frames: {res['non_empty_frames']} ({res['non_empty_frames']/res['total_frames']*100:.1f}%)")
    print(f"  Non-empty frame FPS: {res['non_empty_frames']/TEST_DURATION:.2f}")
    if len(res['delays']) > 1:
        print(f"  Average delay: {np.mean(res['delays'][1:])*1000:.2f} ms")
        print(f"  Minimum delay: {np.min(res['delays'][1:])*1000:.2f} ms")
        print(f"  Maximum delay: {np.max(res['delays'][1:])*1000:.2f} ms")
    else:
        print("  Not enough data for delay statistics")

if __name__ == "__main__":
    print("Testing frame capture performance (polling vs get_frame)")
    print("="*60)
    get_frame_results = test_get_frame()
    time.sleep(1)
    polling_callback_results = test_polling_callback()
    print_summary(get_frame_results)
    print_summary(polling_callback_results)
    plot_results(get_frame_results, polling_callback_results)
    print("\nResults saved to 'capture_performance_comparison.png'") 