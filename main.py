import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
import torchvision.transforms as T
import cv2
import tkinter as tk
from tkinter import filedialog
import time
from fer import FER
import threading

# Load the model with weights
weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
model = model.eval().to('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations
transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
])

# Initialize the emotion detector
emotion_detector = FER()

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Person Detection and Emotion Recognition")
        self.geometry("400x300")
        self.configure(bg="#2c3e50")
        self.create_widgets()

    def create_widgets(self):
        self.real_time_button = tk.Button(self, text="Real-Time Detection", command=self.real_time_detection)
        self.style_button(self.real_time_button)
        
        self.select_image_button = tk.Button(self, text="Select Image", command=self.select_image)
        self.style_button(self.select_image_button)
        
        self.capture_image_button = tk.Button(self, text="Capture Image", command=self.capture_image)
        self.style_button(self.capture_image_button)

        self.help_button = tk.Button(self, text="?", command=self.show_help)
        self.style_help_button(self.help_button)

    def style_button(self, button):
        button.config(bg="#3498db", fg="white", font=("Helvetica", 12, "bold"), activebackground="#2980b9", activeforeground="white", bd=0, highlightthickness=0, relief='flat')
        button.bind("<Enter>", lambda e: button.config(bg="#2980b9"))
        button.bind("<Leave>", lambda e: button.config(bg="#3498db"))
        button.pack(pady=15, padx=20, fill="x", ipady=10)

    def style_help_button(self, button):
        button.config(bg="#e74c3c", fg="white", font=("Helvetica", 12, "bold"), activebackground="#c0392b", activeforeground="white", bd=0, highlightthickness=0, relief='flat', width=2)
        button.bind("<Enter>", lambda e: button.config(bg="#c0392b"))
        button.bind("<Leave>", lambda e: button.config(bg="#e74c3c"))
        button.pack(pady=5, padx=5, anchor='ne')

    def show_help(self):
        help_window = tk.Toplevel(self)
        help_window.title("Help")
        help_window.geometry("400x300")
        help_window.configure(bg="#ecf0f1")
        
        help_text = (
            "How to Use the Application:\n\n"
            "1. Real-Time Detection:\n"
            "   - Click 'Real-Time Detection' to start the webcam.\n"
            "   - The application will detect persons and emotions in real-time.\n"
            "   - Press 'q' to stop the detection.\n\n"
            "2. Select Image:\n"
            "   - Click 'Select Image' to choose an image file from your computer.\n"
            "   - The application will detect persons and emotions in the selected image.\n\n"
            "3. Capture Image:\n"
            "   - Click 'Capture Image' to take a photo using your webcam.\n"
            "   - The application will detect persons and emotions in the captured image.\n\n"
        )

        text_widget = tk.Text(help_window, wrap='word', bg="#ecf0f1", fg="#2c3e50", font=("Helvetica", 10), padx=10, pady=10)
        text_widget.insert('1.0', help_text)
        text_widget.config(state='disabled', width=50)
        text_widget.pack(expand=True, fill='both')

    def real_time_detection(self):
        self.stop_thread = False
        self.cap = cv2.VideoCapture(0)
        self.frame_rate = 30
        self.prev = 0

        self.thread = threading.Thread(target=self.process_frames)
        self.thread.start()

    def process_frames(self):
        while not self.stop_thread:
            time_elapsed = time.time() - self.prev
            ret, frame = self.cap.read()
            if not ret:
                break

            if time_elapsed > 1./self.frame_rate:
                self.prev = time.time()
                original_h, original_w = frame.shape[:2]
                small_frame = cv2.resize(frame, (300, 300))
                resized_h, resized_w = small_frame.shape[:2]

                image_tensor = transform(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
                with torch.no_grad():
                    predictions = model(image_tensor)

                pred_boxes = predictions[0]['boxes'].cpu().numpy()
                pred_labels = predictions[0]['labels'].cpu().numpy()
                pred_scores = predictions[0]['scores'].cpu().numpy()
                for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                    if label == 1 and score > 0.5:
                        x_min, y_min, x_max, y_max = box
                        x_min = int(x_min / resized_w * original_w)
                        y_min = int(y_min / resized_h * original_h)
                        x_max = int(x_max / resized_w * original_w)
                        y_max = int(y_max / resized_h * original_h)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(frame, "Person", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                emotions = emotion_detector.detect_emotions(frame)
                for emotion in emotions:
                    (x, y, w, h) = emotion["box"]
                    emotion_label = emotion["emotions"]
                    main_emotion = max(emotion_label, key=emotion_label.get)
                    cv2.putText(frame, main_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow("Real-Time Detection and Emotion Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_thread = True
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def select_image(self):
        file_path = filedialog.askopenfilename(initialdir="coco/train2017", title="Select Image",
                                               filetypes=(("JPEG files", "*.jpg"), ("all files", "*.*")))
        if file_path:
            self.process_image(file_path)

    def capture_image(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            image_path = "captured_image.jpg"
            cv2.imwrite(image_path, frame)
            self.process_image(image_path)
        cap.release()

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image at {image_path}")
            return
        original_image = image.copy()
        original_h, original_w = original_image.shape[:2]
        small_image = cv2.resize(original_image, (300, 300))
        resized_h, resized_w = small_image.shape[:2]

        image_tensor = transform(cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            predictions = model(image_tensor)

        pred_boxes = predictions[0]['boxes'].cpu().numpy()
        pred_labels = predictions[0]['labels'].cpu().numpy()
        pred_scores = predictions[0]['scores'].cpu().numpy()
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            if label == 1 and score > 0.5:
                x_min, y_min, x_max, y_max = box
                # Scale boxes to original image size
                x_min = int(x_min / resized_w * original_w)
                y_min = int(y_min / resized_h * original_h)
                x_max = int(x_max / resized_w * original_w)
                y_max = int(y_max / resized_h * original_h)
                cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(original_image, "Person", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
        emotions = emotion_detector.detect_emotions(original_image)
        for emotion in emotions:
            (x, y, w, h) = emotion["box"]
            emotion_label = emotion["emotions"]
            main_emotion = max(emotion_label, key=emotion_label.get)
            cv2.putText(original_image, main_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Image Detection and Emotion Recognition", original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = Application()
    app.mainloop()
