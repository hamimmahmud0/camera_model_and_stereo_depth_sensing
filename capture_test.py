import requests
import argparse
import threading
import os
import sys
import termios
import tty
import time
from pathlib import Path
from datetime import datetime

# Global variables
capture_event = threading.Event()
exit_program = threading.Event()
pair_counter = 0
counter_lock = threading.Lock()

def get_single_key():
    """Get a single character from stdin without requiring Enter (Ubuntu/Linux)"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def capture_photo(camera_url, save_path, camera_direction, pair_num):
    """Capture and save a photo from a single camera"""
    try:
        # Send GET request to camera
        response = requests.get(camera_url, timeout=5)
        response.raise_for_status()
        
        # Save the image
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        print(f"[{camera_direction.upper()}] Saved: {save_path} ({len(response.content)} bytes)")
        return True
    except requests.exceptions.RequestException as e:
        print(f"[{camera_direction.upper()}] Error: Failed to capture from {camera_url}: {e}")
        return False
    except Exception as e:
        print(f"[{camera_direction.upper()}] Error: {e}")
        return False

def keyboard_listener():
    """Listen for keyboard input in a separate thread"""
    print("\nPress SPACE to capture simultaneous photos, Q to quit...")
    
    while not exit_program.is_set():
        key = get_single_key()
        
        if key == ' ':  # Space key
            print("\n[SPACE] Capture triggered!")
            capture_event.set()
        elif key.lower() == 'q':  # Q key
            print("\n[Q] Quitting...")
            exit_program.set()
            capture_event.set()  # Also trigger capture event to exit main loop
        else:
            print(f"\n[KEY] Pressed '{key}' - Only SPACE and Q are functional")

def create_save_directory(save_dir):
    """Create save directory if it doesn't exist"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path

def get_next_pair_number(save_dir):
    """Find the next pair number based on existing files in the directory"""
    try:
        files = list(Path(save_dir).glob("*_test_*.jpg"))
        if not files:
            return 0
        
        # Extract pair numbers from filenames
        pair_numbers = []
        for file in files:
            try:
                # Extract number from filename like "l_test_1.jpg"
                base_name = file.stem  # "l_test_1"
                parts = base_name.split('_')
                if len(parts) >= 3 and parts[-1].isdigit():
                    pair_numbers.append(int(parts[-1]))
            except:
                continue
        
        return max(pair_numbers) + 1 if pair_numbers else 0
    except:
        return 0

def start_client(save_dir, l_port, r_port, l_ip, r_ip):
    """Main client function to capture simultaneous photos from two cameras"""
    global pair_counter
    
    # Create save directory
    save_path = create_save_directory(save_dir)
    print(f"Save directory: {save_path.absolute()}")
    
    # Initialize pair counter from existing files
    with counter_lock:
        pair_counter = get_next_pair_number(save_dir)
    print(f"Starting pair counter from: {pair_counter}")
    
    # Camera URLs
    left_camera_url = f"http://{l_ip}:{l_port}/photo.jpg"
    right_camera_url = f"http://{r_ip}:{r_port}/photo.jpg"
    
    print(f"Left camera URL: {left_camera_url}")
    print(f"Right camera URL: {right_camera_url}")
    print("-" * 50)
    
    # Start keyboard listener thread
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()
    
    # Main capture loop
    print("\n=== Camera Calibration Tool ===")
    print("Commands:")
    print("  SPACE - Capture simultaneous photos from both cameras")
    print("  Q     - Quit the program")
    print("=" * 50)
    
    try:
        while not exit_program.is_set():
            # Wait for capture event or exit
            capture_event.wait()
            
            if exit_program.is_set():
                break
            
            # Clear the event for next capture
            capture_event.clear()
            
            # Get current pair number
            with counter_lock:
                current_pair = pair_counter
                pair_counter += 1
            
            # Prepare filenames
            left_filename = save_path / f"l_test_{current_pair}.jpg"
            right_filename = save_path / f"r_test_{current_pair}.jpg"
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Capturing pair #{current_pair}...")
            
            # Create threads for simultaneous capture
            left_thread = threading.Thread(
                target=capture_photo,
                args=(left_camera_url, left_filename, "left", current_pair)
            )
            
            right_thread = threading.Thread(
                target=capture_photo,
                args=(right_camera_url, right_filename, "right", current_pair)
            )
            
            # Start both threads
            left_thread.start()
            right_thread.start()
            
            # Wait for both to complete
            left_thread.join()
            right_thread.join()
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Pair #{current_pair} capture complete!")
            print("\nPress SPACE for next pair, Q to quit...")
            
    except KeyboardInterrupt:
        print("\n\n[Ctrl+C] Interrupted by user")
    finally:
        exit_program.set()
        print("\n" + "=" * 50)
        print(f"Program terminated.")
        print(f"Total pairs captured: {current_pair if 'current_pair' in locals() else 0}")
        print(f"Images saved in: {save_path.absolute()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capture simultaneous images for stereo camera calibration')
    parser.add_argument('--save_dir', type=str, required=True, help='Save Directory Path')
    parser.add_argument('--l_port', type=int, default=8080, help='Left Camera port')
    parser.add_argument('--r_port', type=int, default=8080, help='Right Camera port')
    parser.add_argument('--r_ip', type=str, required=True, help='Right Camera IP address')
    parser.add_argument('--l_ip', type=str, required=True, help='Left Camera IP address')
    
    args = parser.parse_args()
    
    # Check if required packages are installed
    try:
        import requests
    except ImportError:
        print("Error: 'requests' package is not installed.")
        print("Install it with: pip install requests")
        sys.exit(1)
    
    start_client(args.save_dir, args.l_port, args.r_port, args.l_ip, args.r_ip)