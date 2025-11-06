import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
import math

# Fungsi bantu untuk menghitung jarak Euclidean antar titik
def dist(p1, p2):
    return math.dist(p1, p2)

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka. Coba index 1/2.")

# Inisialisasi detektor wajah
detector = FaceMeshDetector(staticMode=False, maxFaces=2,
                            minDetectionCon=0.5, minTrackCon=0.5)

# Variabel untuk menghitung kedipan
blink_count = 0
closed_frames = 0
CLOSED_FRAMES_THRESHOLD = 3   # jumlah frame berturut-turut untuk dianggap kedipan
EYE_AR_THRESHOLD = 0.20       # ambang rasio EAR untuk mata tertutup
is_closed = False

# Indeks landmark mata kiri (mengacu ke model mediapipe face mesh)
L_TOP = 159
L_BOTTOM = 145
L_LEFT = 33
L_RIGHT = 133

while True:
    ok, img = cap.read()
    if not ok:
        break

    img, faces = detector.findFaceMesh(img, draw=True)
    if faces:
        face = faces[0]  # ambil wajah pertama
        v = dist(face[L_TOP], face[L_BOTTOM])  # jarak vertikal mata
        h = dist(face[L_LEFT], face[L_RIGHT])  # jarak horizontal mata
        ear = v / (h + 1e-8)                   # hitung rasio EAR

        # tampilkan EAR
        cv2.putText(img, f"EAR(L): {ear:.3f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # logika kedipan
        if ear < EYE_AR_THRESHOLD:
            closed_frames += 1
            if closed_frames >= CLOSED_FRAMES_THRESHOLD and not is_closed:
                blink_count += 1
                is_closed = True
        else:
            closed_frames = 0
            is_closed = False

        # tampilkan jumlah kedipan
        cv2.putText(img, f"Blink: {blink_count}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("FaceMesh + EAR", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
