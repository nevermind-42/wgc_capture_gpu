import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from mss import mss
import wgc_capture
import pyautogui
from PIL import ImageGrab
import win32gui
import win32ui
import win32con
import win32api
from ctypes import windll

# Test configuration
TEST_DURATION = 10  # seconds
FPS_WINDOW = 1.0  # seconds for sliding FPS window
DISPLAY_FRAMES = '--show' in sys.argv

class CaptureMethod:
    def __init__(self, name):
        self.name = name
        self.frame_count = 0
        self.non_empty_frame_count = 0
        self.delays = []
        self.last_frame_time = None
        self.fps_history = []
        self.current_fps = 0
        self.fps_window = 1.0  # 1 second window for FPS calculation
        self.frame_times = []
        self.capture_times = []  # Время, затраченное на захват кадра

    def get_frame(self):
        raise NotImplementedError

    def cleanup(self):
        pass

    def update_stats(self, frame, capture_time):
        current_time = time.time()
        
        if self.last_frame_time is not None:
            delay = current_time - self.last_frame_time
            self.delays.append(delay)
            self.frame_times.append(current_time)
            self.capture_times.append(capture_time)
            
            # Update FPS using sliding window
            window_start = current_time - self.fps_window
            self.frame_times = [t for t in self.frame_times if t > window_start]
            self.current_fps = len(self.frame_times) / self.fps_window
            self.fps_history.append(self.current_fps)
        
        self.last_frame_time = current_time
        self.frame_count += 1
        
        # Check if frame is non-empty (not all zeros)
        if not np.all(frame == 0):
            self.non_empty_frame_count += 1

class WGCCaptureMethod(CaptureMethod):
    def __init__(self):
        super().__init__("WGC")
        self.capture = wgc_capture.WGCCapture()

    def get_frame(self):
        start_time = time.time()
        frame = self.capture.get_frame()
        capture_time = time.time() - start_time
        self.update_stats(frame, capture_time)
        return frame

    def cleanup(self):
        self.capture = None

class MSSCaptureMethod(CaptureMethod):
    def __init__(self):
        super().__init__("MSS")
        self.sct = mss()

    def get_frame(self):
        start_time = time.time()
        monitor = self.sct.monitors[1]  # Primary monitor
        screenshot = self.sct.grab(monitor)
        frame = np.array(screenshot)
        capture_time = time.time() - start_time
        self.update_stats(frame, capture_time)
        return frame

    def cleanup(self):
        self.sct.close()

class GDICaptureMethod(CaptureMethod):
    def __init__(self):
        super().__init__("GDI")
        self.hwnd = win32gui.GetDesktopWindow()
        self.width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        self.height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)

    def get_frame(self):
        start_time = time.time()
        hwndDC = win32gui.GetWindowDC(self.hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, self.width, self.height)
        saveDC.SelectObject(saveBitMap)
        result = windll.user32.PrintWindow(self.hwnd, saveDC.GetSafeHdc(), 0)
        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        im = np.frombuffer(bmpstr, dtype='uint8')
        im.shape = (self.height, self.width, 4)
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hwndDC)
        capture_time = time.time() - start_time
        self.update_stats(im, capture_time)
        return im

class PyAutoGUICaptureMethod(CaptureMethod):
    def __init__(self):
        super().__init__("PyAutoGUI")
        import pyautogui
        self.pyautogui = pyautogui

    def get_frame(self):
        start_time = time.time()
        screenshot = self.pyautogui.screenshot()
        frame = np.array(screenshot)
        capture_time = time.time() - start_time
        self.update_stats(frame, capture_time)
        return frame

class PILCaptureMethod(CaptureMethod):
    def __init__(self):
        super().__init__("PIL")

    def get_frame(self):
        start_time = time.time()
        screenshot = ImageGrab.grab()
        frame = np.array(screenshot)
        capture_time = time.time() - start_time
        self.update_stats(frame, capture_time)
        return frame

def plot_results(results):
    fig = plt.figure(figsize=(15, 10))
    
    # FPS over time
    plt.subplot(2, 2, 1)
    for method in results:
        plt.plot(method.fps_history, label=method.name)
    plt.title('FPS over time (sliding window)')
    plt.xlabel('Time (frames)')
    plt.ylabel('FPS')
    plt.legend()
    plt.grid(True)
    
    # Capture time distribution
    plt.subplot(2, 2, 2)
    for method in results:
        plt.hist(method.capture_times, bins=50, alpha=0.5, label=method.name)
    plt.title('Frame capture time distribution')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    
    # Average FPS comparison
    plt.subplot(2, 2, 3)
    methods = [m.name for m in results]
    avg_fps = [np.mean(m.fps_history) for m in results]
    plt.bar(methods, avg_fps)
    plt.title('Average FPS comparison')
    plt.xticks(rotation=45)
    plt.ylabel('FPS')
    plt.grid(True)
    
    # Average capture time comparison
    plt.subplot(2, 2, 4)
    avg_capture_times = [np.mean(m.capture_times)*1000 for m in results]  # Convert to ms
    plt.bar(methods, avg_capture_times)
    plt.title('Average capture time comparison')
    plt.xticks(rotation=45)
    plt.ylabel('Time (ms)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('capture_methods_comparison.png')
    plt.close()

def print_summary(results):
    print("\nPerformance Test Results:")
    print("-" * 50)
    for method in results:
        print(f"\n{method.name}:")
        print(f"Total frames: {method.frame_count}")
        print(f"Non-empty frames: {method.non_empty_frame_count}")
        print(f"Non-empty frames percentage: {method.non_empty_frame_count/method.frame_count*100:.2f}%")
        print(f"Average FPS: {np.mean(method.fps_history):.2f}")
        print(f"Min FPS: {np.min(method.fps_history):.2f}")
        print(f"Max FPS: {np.max(method.fps_history):.2f}")
        print(f"Average capture time: {np.mean(method.capture_times)*1000:.2f}ms")
        print(f"Min capture time: {np.min(method.capture_times)*1000:.2f}ms")
        print(f"Max capture time: {np.max(method.capture_times)*1000:.2f}ms")
        print(f"Average delay: {np.mean(method.delays)*1000:.2f}ms")
        print(f"Min delay: {np.min(method.delays)*1000:.2f}ms")
        print(f"Max delay: {np.max(method.delays)*1000:.2f}ms")
        
        # Добавляем информацию о поддержке 3D
        if method.name == "WGC":
            print("\n3D Content Support:")
            print("- DirectX applications")
            print("- OpenGL applications")
            print("- Games")
            print("- Hardware-accelerated video")
            print("- HDR content")
        elif method.name == "MSS":
            print("\n3D Content Support:")
            print("- Basic DirectX support")
            print("- Limited OpenGL support")
            print("- No HDR support")
        elif method.name == "GDI":
            print("\n3D Content Support:")
            print("- No DirectX support")
            print("- No OpenGL support")
            print("- No HDR support")
        elif method.name in ["PyAutoGUI", "PIL"]:
            print("\n3D Content Support:")
            print("- No DirectX support")
            print("- No OpenGL support")
            print("- No HDR support")
            print("- May show black screen for 3D content")

def test_method(method, duration):
    print(f"\nTesting {method.name}...")
    start_time = time.time()
    last_progress_time = start_time
    progress_interval = 1.0  # Show progress every second
    
    try:
        while time.time() - start_time < duration:
            try:
                frame = method.get_frame()
                if frame is not None:
                    current_time = time.time()
                    if current_time - last_progress_time >= progress_interval:
                        elapsed = current_time - start_time
                        progress = (elapsed / duration) * 100
                        print(f"\r{method.name} progress: {progress:.1f}% ({elapsed:.1f}s/{duration}s)", end="")
                        last_progress_time = current_time
                    time.sleep(0.001)  # Small delay to prevent 100% CPU usage
            except Exception as e:
                print(f"\nError in {method.name}: {str(e)}")
                return None
    finally:
        print(f"\n{method.name} test completed. Cleaning up...")
        try:
            method.cleanup()
        except Exception as e:
            print(f"Error cleaning up {method.name}: {str(e)}")
    
    return method

def main():
    # Test duration in seconds for each method
    TEST_DURATION = 10
    
    # Initialize capture methods
    methods = [
        WGCCaptureMethod(),
        MSSCaptureMethod(),
        GDICaptureMethod(),
        PyAutoGUICaptureMethod(),
        PILCaptureMethod()
    ]
    
    print("Starting performance test...")
    print(f"Testing {len(methods)} capture methods:")
    for method in methods:
        print(f"- {method.name}")
    print("\nTest duration per method:", TEST_DURATION, "seconds")
    print("\nNote: For best results with 3D content:")
    print("- WGC: Full support for DirectX, OpenGL, and HDR")
    print("- MSS: Basic support for DirectX")
    print("- Other methods: May show black screen for 3D content")
    print("-" * 50)
    
    # Test each method separately
    working_methods = []
    for i, method in enumerate(methods, 1):
        print(f"\nMethod {i}/{len(methods)}")
        result = test_method(method, TEST_DURATION)
        if result is not None:
            working_methods.append(result)
            print(f"{method.name} test successful")
        else:
            print(f"{method.name} test failed")
    
    if not working_methods:
        print("\nNo methods completed successfully!")
        return
        
    print("\nTest completed. Processing results...")
    
    # Plot results
    try:
        plot_results(working_methods)
        print("Results plotted successfully")
    except Exception as e:
        print(f"Error plotting results: {str(e)}")
    
    # Print summary
    try:
        print_summary(working_methods)
    except Exception as e:
        print(f"Error printing summary: {str(e)}")
    
    print("\nResults saved to 'capture_methods_comparison.png'")

if __name__ == "__main__":
    main() 