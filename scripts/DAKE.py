import os
import cv2
import numpy as np
from collections import deque
def extract_keyframes_perfect_balanced(
    video_path, 
    k_global=2,            # Độ nhạy cắt cảnh toàn phần (Tăng nhẹ lên 2.5 để thưa hơn TransNet)
    local_threshold_static=10.0, # 🌟 HẠ THẤP: Rất nhạy để bắt trọn thông tin nhỏ trong query khi bảng tĩnh
    min_frame_gap_action=5, # Khoảng cách frame khi có hành động (~0.7s) -> Đảm bảo thưa hơn TransNet
    min_frame_gap_static=45, # Khoảng cách frame khi bảng tĩnh (~1.5s) -> Chờ chữ viết xong
    window_size=60, # Kích thước cửa sổ trượt để tính toán biến động toàn cảnh (1-2s)
    output_dir="multimedia_system_index"
):
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0
    
    global_delta_window = deque(maxlen=window_size)
    prev_gray = None
    prev_jpeg_size = None
    last_saved_gray = None 
    
    frame_count = 0
    shots_detected = 0
    last_keyframe_frame = -min_frame_gap_static
    grid_rows, grid_cols = 4, 4

    print(f"Analyzing with Decoupled Hybrid Engine... Saving to '{output_dir}/'")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        success, encoded_image = cv2.imencode('.jpg', frame)
        if not success:
            continue
        current_jpeg_size = len(encoded_image.tobytes())
        
        # Mặc định trích xuất frame đầu tiên
        if prev_gray is None:
            time_sec = 0.0
            cv2.imwrite(os.path.join(output_dir, f"shot_{shots_detected:04d}_{time_sec:.2f}s.jpg"), frame)
            prev_gray = gray
            last_saved_gray = gray
            prev_jpeg_size = current_jpeg_size
            last_keyframe_frame = frame_count
            shots_detected += 1
            frame_count += 1
            continue
            
        # 1. LUỒNG TOÀN CẢNH (GLOBAL DELTA)
        global_delta = abs(current_jpeg_size - prev_jpeg_size)
        
        if len(global_delta_window) < window_size:
            global_delta_window.append(global_delta)
            prev_gray = gray
            prev_jpeg_size = current_jpeg_size
            frame_count += 1
            continue
            
        # Tính toán nền nhiễu động
        local_mean = np.mean(global_delta_window)
        local_std = np.std(global_delta_window)
        noise_floor = max(current_jpeg_size * 0.012, 2500.0)
        
        # 🌟 PHÂN LOẠI TRẠNG THÁI VIDEO CHÍNH XÁC
        # Nếu biến động byte nhỏ hơn sàn nhiễu -> Video đang ở trạng thái TĨNH (Bài giảng/Slide)
        is_video_static = global_delta < noise_floor
        
        # ĐIỀU KIỆN KÍCH HOẠT ĐỘC LẬP THEO TRẠNG THÁI
        trigger_fired = False
        trigger_type = ""
        
        # THỬ NGHIỆM COOLDOWN THEO TRẠNG THÁI
        current_gap = min_frame_gap_static if is_video_static else min_frame_gap_action
        is_outside_cooldown = (frame_count - last_keyframe_frame) >= current_gap
        
        if is_outside_cooldown:
            if not is_video_static:
                # 🎬 TRẠNG THÁI HÀNH ĐỘNG/NGOẠI CẢNH: Chỉ dùng DAKE để bắt cut cảnh, KHÓA CHẶT LUỒNG GRID
                adaptive_global_threshold = local_mean + (k_global * max(local_std, noise_floor / k_global))
                if global_delta > adaptive_global_threshold:
                    trigger_fired = True
                    trigger_type = "GLOBAL_CUT_ACTION"
            else:
                # 📝 TRẠNG THÁI TĨNH/BÀI GIẢNG: Bật lưới với độ nhạy cực cao để rà thông tin query
                content_diff = cv2.absdiff(gray, last_saved_gray)
                h, w = content_diff.shape
                row_step, col_step = h // grid_rows, w // grid_cols
                max_grid_change = 0
                
                for r in range(grid_rows):
                    for c in range(grid_cols):
                        roi = content_diff[r*row_step:(r+1)*row_step, c*col_step:(c+1)*col_step]
                        grid_mean = np.mean(roi)
                        if grid_mean > max_grid_change:
                            max_grid_change = grid_mean
                
                if max_grid_change > local_threshold_static:
                    trigger_fired = True
                    trigger_type = "LOCAL_TEXT_STATIC"

        # THỰC THI LƯU FRAME
        if trigger_fired:
            time_sec = frame_count / fps
            out_path = os.path.join(output_dir, f"shot_{shots_detected:04d}_{time_sec:.2f}s.jpg")
            cv2.imwrite(out_path, frame)
            
            shots_detected += 1
            last_keyframe_frame = frame_count
            last_saved_gray = gray # Cập nhật mốc tham chiếu
            
        global_delta_window.append(global_delta)
        prev_gray = gray
        prev_jpeg_size = current_jpeg_size
        frame_count += 1

    cap.release()
    print(f"\nHoàn thành cấu hình tối ưu: Trích xuất {shots_detected} keyframes.")
    return shots_detected
if __name__ == "__main__":
    extract_keyframes_perfect_balanced(
        video_path="sample_video.mp4",
    )
