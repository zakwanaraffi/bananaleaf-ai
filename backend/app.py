from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, sqlite3, shutil, time
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

# ===============================
# CONFIG
# ===============================
BASE_DIR            = Path(__file__).parent.resolve()
FRONTEND_DIR        = str(BASE_DIR.parent / "frontend")
UPLOAD_FOLDER       = str(BASE_DIR / "uploads")
SAVED_IMAGES_FOLDER = str(BASE_DIR / "saved_images")
os.makedirs(UPLOAD_FOLDER,       exist_ok=True)
os.makedirs(SAVED_IMAGES_FOLDER, exist_ok=True)

DB_PATH    = str(BASE_DIR / "history.db")
MODEL_PATH = str(BASE_DIR / "models" / "best.pt")

model = YOLO(MODEL_PATH)
print(f"[INFO] Model task  : {model.task}")
print(f"[INFO] Model classes: {model.names}")

# Path COCO model: cari di models/ (hosting) atau parent dir (lokal)
_coco_in_models = BASE_DIR / "models" / "yolov8n.pt"
_coco_in_parent = BASE_DIR.parent / "yolov8n.pt"
if _coco_in_models.exists():
    COCO_MODEL_PATH = str(_coco_in_models)
else:
    COCO_MODEL_PATH = str(_coco_in_parent)
print(f"[INFO] COCO model path: {COCO_MODEL_PATH}")
coco_model = YOLO(COCO_MODEL_PATH)

IS_DETECTION = (model.task == "detect")
print(f"[INFO] Mode: {'OBJECT DETECTION' if IS_DETECTION else 'CLASSIFICATION'}")


# ===============================
# DATABASE
# ===============================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_name  TEXT NOT NULL,
                class_name  TEXT NOT NULL,
                confidence  REAL NOT NULL,
                solution    TEXT NOT NULL,
                image_path  TEXT,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        try:
            conn.execute("ALTER TABLE history ADD COLUMN image_path TEXT")
        except Exception:
            pass
        conn.commit()

init_db()


# ===============================
# SOLUTIONS
# ===============================
SOLUTION_SIGATOKA = (
    "Daun terdeteksi terinfeksi penyakit Sigatoka yang disebabkan oleh jamur "
    "Mycosphaerella musicola (Sigatoka Kuning) atau Pseudocercospora fijiensis (Sigatoka Hitam). "
    "Berdasarkan kajian FAO dan CIRAD, penyakit ini dapat menyebabkan penurunan hasil panen "
    "hingga 30-50% apabila tidak ditangani, dengan potensi kerugian total hingga 100% pada "
    "kasus terparah. Selain mengurangi bobot tandan, infeksi yang berat terbukti secara "
    "ilmiah mempersingkat masa simpan buah (green life), yaitu mempercepat proses "
    "pematangan buah secara prematur akibat peningkatan produksi etilen dan CO2 pada "
    "puncak klimakterik buah. Hal ini menyebabkan buah pisang tidak layak ekspor karena "
    "matang sebelum sampai ke tujuan. "
    "Tindakan Penanganan: Segera potong dan musnahkan daun yang terinfeksi berat "
    "agar spora tidak menyebar ke tanaman sekitarnya. "
    "Semprotkan fungisida golongan strobilurin seperti trifloxystrobin atau "
    "pyraclostrobin yang terbukti paling efektif berdasarkan penelitian lapangan. "
    "Sebagai alternatif, fungisida triazol seperti propiconazole atau epoxiconazole "
    "juga dapat digunakan. Lakukan rotasi fungisida dengan mancozeb setiap 14 hingga "
    "21 hari sebagai protektan dan untuk mencegah berkembangnya resistensi. "
    "Pastikan tanaman memiliki minimal 5 daun sehat saat panen untuk menjamin "
    "bobot tandan yang optimal. Pantau perkembangan penyakit secara berkala."
)

SOLUTION_LEAFSPOT = (
    "Daun terdeteksi mengalami bercak daun (Leaf Spot) yang disebabkan oleh infeksi "
    "jamur Neocordana musae (Cordana musae). Berdasarkan penelitian dari Universitas "
    "Muhammadiyah Malang dan Institut Pertanian Bogor, bercak nekrotik yang terbentuk "
    "pada daun secara langsung mengurangi luas area fotosintesis aktif tanaman. "
    "Penurunan kapasitas fotosintesis ini berdampak pada berkurangnya cadangan energi "
    "yang dibutuhkan untuk pengisian buah, sehingga mengakibatkan ukuran buah yang "
    "lebih kecil, jumlah buah per tandan yang lebih sedikit, serta kualitas fisik buah "
    "yang tidak memenuhi standar pasar atau kriteria ekspor. Pada infeksi berat, "
    "kerugian ekonomi yang ditimbulkan dapat signifikan bagi petani. "
    "Tindakan Penanganan: Potong dan buang daun yang telah menunjukkan gejala bercak "
    "untuk mencegah penyebaran spora ke daun dan tanaman sehat di sekitarnya. "
    "Aplikasikan fungisida protektan berbahan aktif mancozeb secara rutin "
    "setiap 14 hingga 21 hari sebagai penanganan utama. "
    "Pada serangan yang lebih parah, kombinasikan dengan propiconazole "
    "untuk meningkatkan efektivitas pengendalian. "
    "Pastikan drainase kebun berjalan baik dan hindari percikan air antar tanaman "
    "karena kelembapan dan suhu hangat merupakan faktor utama penyebaran spora jamur ini."
)

SOLUTION_HEALTHY = (
    "Daun pisang dalam kondisi sehat dan tidak terdeteksi gejala penyakit. "
    "Kondisi daun yang sehat sangat penting dijaga karena berdasarkan penelitian "
    "lapangan, bobot tandan buah pisang sangat bergantung pada jumlah daun sehat "
    "yang tersisa saat panen. Tanaman dengan kurang dari 5 daun sehat pada saat "
    "panen umumnya menghasilkan tandan yang lebih ringan dan kualitas buah yang "
    "lebih rendah. "
    "Saran Pemeliharaan: Pertahankan kondisi ini dengan menjaga pemupukan yang "
    "seimbang (N-P-K) dan memastikan drainase lahan berjalan baik untuk mencegah "
    "kelembapan berlebih yang menjadi media tumbuh jamur penyebab penyakit Sigatoka "
    "maupun Bercak Daun. Lakukan pemantauan visual pada daun secara rutin setiap "
    "minggu agar gejala penyakit dapat terdeteksi dan ditangani sejak dini sebelum "
    "berdampak pada kualitas buah."
)

SOLUTION_UNKNOWN = (
    "Tidak ada daun pisang terdeteksi dalam gambar. "
    "Pastikan foto yang diunggah menampilkan daun pisang secara jelas dengan "
    "pencahayaan yang cukup dan fokus yang baik. "
    "Hindari latar belakang yang terlalu kompleks dan pastikan daun pisang "
    "menjadi objek utama dalam foto, kemudian coba unggah kembali."
)

SOLUTION_MAP = {
    "healthy":  SOLUTION_HEALTHY,
    "sigatoka": SOLUTION_SIGATOKA,
    "leafspot": SOLUTION_LEAFSPOT,
}

# ===============================
# THRESHOLDS
# ===============================
MIN_GREEN_RATIO_NORMAL  = 0.25
MIN_GREEN_RATIO_DARK    = 0.10   

DARK_IMAGE_THRESHOLD    = 140    
DARK_ENHANCE_MAX_FACTOR = 3.5

DETECTION_CONF_THRESHOLD = 0.35  
DETECTION_IOU_THRESHOLD  = 0.30  


# ===============================
# PREPROCESSING
# ===============================
def preprocess_image(img):
    img_array = np.array(img)   
    avg_brightness = np.mean(img_array)
    was_enhanced = False

    print(f"  [PRE] avg_brightness={avg_brightness:.1f}")

    if avg_brightness < DARK_IMAGE_THRESHOLD:
        was_enhanced = True
        img_bgr  = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_lab  = cv2.cvtColor(img_bgr,   cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img_lab)
        clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        img_lab_enhanced = cv2.merge([l_clahe, a, b])
        img_bgr_enhanced = cv2.cvtColor(img_lab_enhanced, cv2.COLOR_LAB2BGR)
        img_rgb_enhanced = cv2.cvtColor(img_bgr_enhanced, cv2.COLOR_BGR2RGB)
        avg_after_clahe = np.mean(img_rgb_enhanced)
        if avg_after_clahe < 100:
            factor = min(140 / max(avg_after_clahe, 1), DARK_ENHANCE_MAX_FACTOR)
            img_rgb_enhanced = np.clip(
                img_rgb_enhanced.astype(float) * factor, 0, 255
            ).astype(np.uint8)
            print(f"  [PRE] Extra boost: {avg_after_clahe:.1f} -> "
                  f"{np.mean(img_rgb_enhanced):.1f} (x{factor:.2f})")

        avg_final = np.mean(img_rgb_enhanced)
        print(f"  [PRE] CLAHE done: {avg_brightness:.1f} -> {avg_final:.1f}")
        img_array = img_rgb_enhanced
        kernel_gentle = np.array([[ 0, -0.5,  0],
                                  [-0.5,  3, -0.5],
                                  [ 0, -0.5,  0]])
        img_array = cv2.filter2D(img_array, -1, kernel_gentle)
        print("  [PRE] Gentle sharpening applied (dark image only).")
        return Image.fromarray(img_array), True

    return img, False


# ===============================
# COLOR ANALYSIS
# ===============================
def analyze_crop_color(img_array, bbox_xyxy):
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    h, w = img_array.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1: return None
    crop = img_array[y1:y2, x1:x2]
    if crop.size == 0: return None
    ch, cw = crop.shape[:2]
    if max(ch, cw) > 100:
        step = max(1, max(ch, cw) // 100)
        crop = crop[::step, ::step]

    r = crop[:, :, 0].astype(float)
    g = crop[:, :, 1].astype(float)
    b = crop[:, :, 2].astype(float)
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    diff = maxc - minc
    hue = np.zeros_like(maxc)
    mg = (maxc == g) & (diff > 0)
    mr = (maxc == r) & (diff > 0)
    mb = (maxc == b) & (diff > 0)
    hue[mr] = (60 * ((g[mr] - b[mr]) / diff[mr]) + 360) % 360
    hue[mg] = (60 * ((b[mg] - r[mg]) / diff[mg]) + 120) % 360
    hue[mb] = (60 * ((r[mb] - g[mb]) / diff[mb]) + 240) % 360
    sat = np.zeros_like(maxc)
    sat[maxc > 0] = (diff[maxc > 0] / maxc[maxc > 0]) * 255
    val = maxc
    total_pixels = hue.size
    if total_pixels == 0: return None

    green_pixels  = np.sum((hue >= 60)  & (hue <= 160) & (sat > 60) & (val > 30))
    brown_pixels  = np.sum((hue >= 15)  & (hue <= 45)  & (sat > 40) & (val > 30))
    yellow_pixels = np.sum((hue >= 46)  & (hue <= 59)  & (sat > 40) & (val > 30))

    green_ratio     = green_pixels  / total_pixels
    brown_ratio     = brown_pixels  / total_pixels
    yellow_ratio    = yellow_pixels / total_pixels
    non_green_ratio = 1.0 - green_ratio

    return {
        "green_ratio":     green_ratio,
        "brown_ratio":     brown_ratio,
        "yellow_ratio":    yellow_ratio,
        "non_green_ratio": non_green_ratio,
    }

# ===============================
# CASSAVA LEAF DETECTOR
# ===============================
def is_cassava_background(img_array, img_width, img_height):
    h, w = img_array.shape[:2]
    corner_size_h = max(1, h // 10)
    corner_size_w = max(1, w // 10)
    corners = [
        img_array[:corner_size_h, :corner_size_w],
        img_array[:corner_size_h, w-corner_size_w:],
        img_array[h-corner_size_h:, :corner_size_w],
        img_array[h-corner_size_h:, w-corner_size_w:],
    ]
    bright_corner_count = 0
    for corner in corners:
        avg_brightness = np.mean(corner)
        r_mean = np.mean(corner[:,:,0])
        g_mean = np.mean(corner[:,:,1])
        b_mean = np.mean(corner[:,:,2])
        is_bright   = avg_brightness > 170
        is_neutral  = abs(float(r_mean) - float(g_mean)) < 30 and abs(float(g_mean) - float(b_mean)) < 30
        is_wood     = float(r_mean) > float(g_mean) * 0.9 and float(r_mean) > 140
        if is_bright and (is_neutral or is_wood): bright_corner_count += 1
    is_studio = bright_corner_count >= 3
    print(f"  [BG CHECK] bright_corners={bright_corner_count}/4, is_studio={is_studio}")
    return is_studio

# ===============================
# CROP VALIDATION
# ===============================
def is_leaf_crop(img_array, bbox_xyxy, img_width, img_height, was_enhanced=False):
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    h, w = img_array.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1: return False, "invalid_box"

    # --- Color analysis ---
    color_stats = analyze_crop_color(img_array, bbox_xyxy)
    if color_stats is None: return True, "color_analysis_failed_but_passed"
    green_ratio     = color_stats["green_ratio"]
    brown_ratio     = color_stats["brown_ratio"]
    yellow_ratio    = color_stats["yellow_ratio"]
    non_green_ratio = color_stats["non_green_ratio"]
    print(f"  [COLOR] green={green_ratio:.2f}, brown={brown_ratio:.2f}, yellow={yellow_ratio:.2f}, non_green={non_green_ratio:.2f}, enhanced={was_enhanced}")
    min_green = MIN_GREEN_RATIO_DARK if was_enhanced else MIN_GREEN_RATIO_NORMAL
    if green_ratio < min_green:
        return False, f"low_green_ratio({green_ratio:.2f}<{min_green:.2f})"
    total_leaf_color = green_ratio + brown_ratio + yellow_ratio
    if total_leaf_color < 0.10: return False, f"not_a_leaf(colors={total_leaf_color:.2f})"
    if not was_enhanced:
        if is_cassava_background(img_array, img_width, img_height): return False, "studio_background_detected"
    return color_stats, f"ok(green={green_ratio:.2f})"

# ===============================
# BOX MERGE
# ===============================
def merge_overlapping_boxes(det_list, overlap_threshold=0.50):
    if len(det_list) <= 1: return det_list
    merged = []
    used = [False] * len(det_list)
    for i in range(len(det_list)):
        if used[i]: continue
        keep_idx = i
        for j in range(i + 1, len(det_list)):
            if used[j]: continue
            b1 = det_list[keep_idx]["bbox"]
            b2 = det_list[j]["bbox"]
            x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1]); x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            min_area = min(area1, area2)
            iom = inter / min_area if min_area > 0 else 0
            if iom > overlap_threshold:
                cls1 = det_list[keep_idx]["class_name"].lower()
                cls2 = det_list[j]["class_name"].lower()
                conf1 = det_list[keep_idx]["confidence"]
                conf2 = det_list[j]["confidence"]
                
                if cls1 == "healthy" and cls2 in ["leafspot", "sigatoka"]:
                    if conf2 >= 40.0 or conf2 >= conf1 - 15.0:
                        used[keep_idx] = True; keep_idx = j
                    else:
                        used[j] = True
                elif cls2 == "healthy" and cls1 in ["leafspot", "sigatoka"]:
                    if conf1 >= 40.0 or conf1 >= conf2 - 15.0:
                        used[j] = True
                    else:
                        used[keep_idx] = True; keep_idx = j
                else:
                    if conf2 > conf1:
                        used[keep_idx] = True; keep_idx = j
                    else:
                        used[j] = True
                print(f"  [MERGE] Resolved overlapping box between {cls1} and {cls2}")
        merged.append(det_list[keep_idx])
        used[keep_idx] = True
    return merged

# ===============================
# MULTI-SCALE PREDICT
# ===============================
def predict_multiscale(img_pil, was_enhanced):
    best_result   = None
    best_max_conf = 0.0
    best_scale    = 640
    scales = [640, 800, 1024]
    for scale in scales:
        results = model(img_pil, imgsz=scale, conf=DETECTION_CONF_THRESHOLD, iou=DETECTION_IOU_THRESHOLD, agnostic_nms=True, max_det=10)
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            max_conf = float(max(b.conf[0] for b in boxes))
            print(f"  [SCALE={scale}] detections={len(boxes)}, max_conf={max_conf:.2f}")
            if max_conf > best_max_conf:
                best_max_conf = max_conf
                best_result   = results[0]
                best_scale    = scale
        else:
            print(f"  [SCALE={scale}] no detections")
    print(f"  [MULTISCALE] best_scale={best_scale}, max_conf={best_max_conf:.2f}")

    if best_result is None: best_result = model(img_pil, imgsz=640, conf=DETECTION_CONF_THRESHOLD)[0]
    return best_result

# ===============================
# SERVE FRONTEND
# ===============================
@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/detect")
def serve_detect():
    return send_from_directory(FRONTEND_DIR, "detect.html")

@app.route("/history-page")
def serve_history_page():
    return send_from_directory(FRONTEND_DIR, "history.html")

@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(str(Path(FRONTEND_DIR) / "css"), filename)

# ===============================
# SERVE IMAGES
# ===============================
@app.route("/images/<path:filename>")
def serve_image(filename): return send_from_directory(SAVED_IMAGES_FOLDER, filename)

# ===============================
# PREDICT
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files: return jsonify({"error": "No image uploaded"}), 400
        file = request.files["image"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        # COCO person detection filter (reject selfies/human presence)
        coco_results = coco_model(filepath, conf=0.35, verbose=False)
        coco_boxes = coco_results[0].boxes
        if coco_boxes is not None:
            orig_h, orig_w = coco_results[0].orig_shape
            img_area = orig_h * orig_w
            reject_person = False
            for box in coco_boxes:
                cls_id = int(box.cls[0])
                if coco_model.names[cls_id] == "person":
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    box_area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                    area_pct = (box_area / img_area) * 100
                    
                    if conf >= 0.65 or (conf >= 0.35 and area_pct >= 15.0):
                        print(f"[REJECT PERSON] file={file.filename}, conf={conf:.2f}, area={area_pct:.1f}%")
                        reject_person = True
                        break
            
            if reject_person:
                return jsonify({
                    "class_name": "unknown",
                    "confidence": 0,
                    "detections": [],
                    "solution": SOLUTION_UNKNOWN,
                    "temp_image": file.filename,
                    "img_width": orig_w,
                    "img_height": orig_h
                })
        
        img = Image.open(filepath).convert("RGB")
        img_for_predict, was_enhanced = preprocess_image(img)
        if IS_DETECTION:
            det_list = []
            w_img, h_img = img.size
            
            # Pass 1: Raw image
            best_result_raw = predict_multiscale(img, False)
            img_array_raw = np.array(img)
            if best_result_raw.boxes is not None and len(best_result_raw.boxes) > 0:
                for box in best_result_raw.boxes:
                    cls_id   = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    conf     = round(float(box.conf[0]) * 100, 2)
                    xyxy     = box.xyxy[0].tolist()
                    color_stats_or_bool, reason_str = is_leaf_crop(img_array_raw, xyxy, w_img, h_img, False)
                    if not color_stats_or_bool:
                        print(f"  [RAW CROP REJECTED] cls={cls_name}, conf={conf}% -> {reason_str}")
                        continue
                    bbox_norm = [xyxy[0] / w_img, xyxy[1] / h_img, xyxy[2] / w_img, xyxy[3] / h_img]
                    det_list.append({"class_name": cls_name, "confidence": conf, "bbox": bbox_norm})
                    print(f"  [RAW CROP ACCEPTED] cls={cls_name}, conf={conf}% -> {reason_str}")

            # Pass 2: Preprocessed image (only if enhancement was applied)
            if was_enhanced:
                best_result_prep = predict_multiscale(img_for_predict, True)
                img_array_prep = np.array(img_for_predict)
                if best_result_prep.boxes is not None and len(best_result_prep.boxes) > 0:
                    for box in best_result_prep.boxes:
                        cls_id   = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        conf     = round(float(box.conf[0]) * 100, 2)
                        xyxy     = box.xyxy[0].tolist()
                        color_stats_or_bool, reason_str = is_leaf_crop(img_array_prep, xyxy, w_img, h_img, True)
                        if not color_stats_or_bool:
                            print(f"  [PREP CROP REJECTED] cls={cls_name}, conf={conf}% -> {reason_str}")
                            continue
                        bbox_norm = [xyxy[0] / w_img, xyxy[1] / h_img, xyxy[2] / w_img, xyxy[3] / h_img]
                        det_list.append({"class_name": cls_name, "confidence": conf, "bbox": bbox_norm})
                        print(f"  [PREP CROP ACCEPTED] cls={cls_name}, conf={conf}% -> {reason_str}")

            # Merge overlapping detections
            det_list = merge_overlapping_boxes(det_list)
            
            # Confidence boosting rule: C_new = C + (100 - C) * 0.4
            for d in det_list:
                d["confidence"] = round(d["confidence"] + (100 - d["confidence"]) * 0.4, 2)
            if not det_list:
                print(f"[NO DETECTION] file={file.filename}")
                return jsonify({"class_name": "unknown", "confidence": 0, "detections": [], "solution": SOLUTION_UNKNOWN, "temp_image": file.filename, "img_width": img.width, "img_height": img.height})
            best = max(det_list, key=lambda d: d["confidence"])
            dominant = best["class_name"]
            solution = SOLUTION_MAP.get(dominant.lower(), "Kelas tidak dikenali.")
            print(f"[DETECT OK] file={file.filename}, dominant={dominant}, total_det={len(det_list)}, enhanced={was_enhanced}")
            return jsonify({"class_name": dominant, "confidence": best["confidence"], "detections": det_list, "solution": solution, "temp_image": file.filename, "img_width": img.width, "img_height": img.height})
        else:
            probs = model(img_for_predict)[0].probs.data.tolist()
            class_id = probs.index(max(probs))
            class_name = model.names[class_id]
            conf_pct = round(max(probs) * 100, 2)
            solution = SOLUTION_MAP.get(class_name.lower(), "Kelas tidak dikenali.")
            return jsonify({"class_name": class_name, "confidence": conf_pct, "detections": [{"class_name": class_name, "confidence": conf_pct, "bbox": [0,0,1,1]}], "solution": solution, "temp_image": file.filename, "img_width": img.width, "img_height": img.height})
    except Exception as e:
        print("PREDICT ERROR:", e)
        return jsonify({"class_name": "error", "confidence": 0, "solution": "Terjadi error pada server."}), 500

# ===============================
# SAVE RESULT
# ===============================
@app.route("/save", methods=["POST"])
def save():
    try:
        data = request.get_json()
        owner_name = data.get("owner_name", "").strip()
        class_name = data.get("class_name", "")
        confidence = data.get("confidence", 0)
        solution   = data.get("solution", "")
        temp_image = data.get("temp_image", "")
        if not owner_name: return jsonify({"error": "Nama tidak boleh kosong."}), 400
        with get_db() as conn:
            saved_image_name = None
            if temp_image:
                src = os.path.join(UPLOAD_FOLDER, temp_image)
                if os.path.exists(src):
                    ext  = os.path.splitext(temp_image)[1] or ".jpg"
                    safe = owner_name.replace(" ", "_").replace("/", "_")
                    saved_image_name = f"{safe}_{int(time.time())}{ext}"
                    shutil.copy2(src, os.path.join(SAVED_IMAGES_FOLDER, saved_image_name))
            conn.execute("INSERT INTO history (owner_name, class_name, confidence, solution, image_path) VALUES (?, ?, ?, ?, ?)", (owner_name, class_name, confidence, solution, saved_image_name))
            conn.commit()
        return jsonify({"message": "Berhasil disimpan!"}), 200
    except Exception as e: return jsonify({"error": "Terjadi error pada server."}), 500

# ===============================
# HISTORY
# ===============================
@app.route("/history", methods=["GET"])
def get_history():
    try:
        with get_db() as conn: rows = conn.execute("SELECT * FROM history ORDER BY created_at DESC").fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["image_url"] = f"{request.host_url}images/{item['image_path']}" if item.get("image_path") else None
            result.append(item)
        return jsonify(result), 200
    except Exception as e:
        print("GET HISTORY ERROR:", e)
        return jsonify({"error": "Terjadi error pada server."}), 500

@app.route("/history/<int:item_id>", methods=["GET"])
def get_history_detail(item_id):
    try:
        with get_db() as conn: row = conn.execute("SELECT * FROM history WHERE id = ?", (item_id,)).fetchone()
        if not row: return jsonify({"error": "Data tidak ditemukan"}), 404
        item = dict(row)
        item["image_url"] = f"{request.host_url}images/{item['image_path']}" if item.get("image_path") else None
        return jsonify(item), 200
    except Exception: return jsonify({"error": "Terjadi error pada server."}), 500

@app.route("/history/<int:item_id>", methods=["DELETE"])
def delete_history(item_id):
    try:
        with get_db() as conn:
            row = conn.execute("SELECT image_path FROM history WHERE id = ?", (item_id,)).fetchone()
            if not row: return jsonify({"error": "Data tidak ditemukan"}), 404
            if row["image_path"]:
                p = os.path.join(SAVED_IMAGES_FOLDER, row["image_path"])
                if os.path.exists(p): os.remove(p)
            conn.execute("DELETE FROM history WHERE id = ?", (item_id,))
            conn.commit()
        return jsonify({"message": "Berhasil dihapus"}), 200
    except Exception: return jsonify({"error": "Terjadi error pada server."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
