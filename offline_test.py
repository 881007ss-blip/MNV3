
import cv2
import numpy as np
import torch
import time
import csv
from PIL import Image
from collections import deque
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small

device = "cuda" if torch.cuda.is_available() else "cpu"
SEQUENCE_LENGTH = 15
CONFIDENCE_THRESHOLD = 0.60
actions = ["baseball", "jump", "kick", "punch", "squat", "idle", "swing"]

class MobileNetV3Feature(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = mobilenet_v3_small(pretrained=True)
        self.features = nn.Sequential(
            *list(backbone.children())[:-2],
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(576 * 2 * 2, 128)
        )

    def forward(self, x):
        return self.features(x)

class ActionRecognitionModel(nn.Module):
    def __init__(self, feature_extractor, num_classes=7):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.temporal = nn.GRU(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=False
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        B, T = x.shape[:2]
        x = x.view(B * T, *x.shape[2:])
        spatial_feat = self.feature_extractor(x)
        temporal_in = spatial_feat.view(B, T, -1)
        temporal_out, _ = self.temporal(temporal_in)
        return self.classifier(temporal_out.mean(1))

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


model = ActionRecognitionModel(MobileNetV3Feature())
checkpoint = torch.load("your_model.pth", map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()


def preprocess_frame(frame):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return preprocess(img_pil)

def predict_action(frame_queue):
    sequence = torch.stack(list(frame_queue)).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(sequence)
        probabilities = F.softmax(predictions, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        return predicted_class.item(), confidence.item(), probabilities.cpu().numpy()[0]


def main(video_path):
    text_params = {
        'y_start': 50,
        'x_offset': 30,
        'font_scale': 1.2,
        'font_thickness': 3,
        'line_spacing': 60,
        'bg_padding': 10,
        'bg_color': (50, 50, 50),
        'text_color': (255, 255, 255),
        'idle_color': (100, 100, 255),
        'low_confidence_color': (100, 100, 100)
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Can't open: {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    recording_time = time.strftime("%Y%m%d_%H%M%S")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        f'output_{recording_time}.mp4',
        fourcc,
        original_fps,
        (1280, 720)
    )

    log_path = f'output_log_{recording_time}.csv'
    log_file = open(log_path, mode='w', newline='', encoding='utf-8')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['frame_id', 'timestamp', 'action', 'confidence'] + actions)

    frame_queue = deque(maxlen=SEQUENCE_LENGTH)
    time_window = deque(maxlen=30)

    while cap.isOpened():
        loop_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        processed = preprocess_frame(frame)
        frame_queue.append(processed)

        display_frame = cv2.resize(frame, (1280, 720))  

        if len(frame_queue) == SEQUENCE_LENGTH:
            action_idx, confidence, probs = predict_action(frame_queue)
            torch.cuda.synchronize() if device == "cuda" else None
            max_prob_idx = np.argmax(probs)

            if confidence >= CONFIDENCE_THRESHOLD:
                display_text = actions[action_idx]
                main_color = (0, 255, 255)
            else:
                display_text = "idle"
                main_color = text_params['idle_color']
                action_idx = 5  

            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            log_writer.writerow([
                frame_id, f"{timestamp:.2f}", display_text, f"{confidence:.4f}"
            ] + [f"{p:.4f}" for p in probs])

            main_text = f"{display_text} ({confidence * 100:.1f}%)"
            (text_width, text_height), _ = cv2.getTextSize(
                main_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                text_params['font_scale'],
                text_params['font_thickness']
            )
            x_main = display_frame.shape[1] - text_width - 50
            y_main = text_params['y_start'] + text_height
            cv2.rectangle(
                display_frame,
                (x_main - text_params['bg_padding'], y_main - text_height - text_params['bg_padding']),
                (x_main + text_width + text_params['bg_padding'], y_main + text_params['bg_padding']),
                text_params['bg_color'],
                -1
            )
            cv2.putText(
                display_frame, main_text,
                (x_main, y_main),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_params['font_scale'],
                main_color,
                text_params['font_thickness']
            )

            for i, (action, prob) in enumerate(zip(actions, probs)):
                color = (text_params['low_confidence_color']
                         if confidence < CONFIDENCE_THRESHOLD
                         else (0, 0, 255) if i == max_prob_idx else (0, 255, 0))
                text = f"{action}: {prob * 100:.1f}%"
                (tw, th), _ = cv2.getTextSize(
                    text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    text_params['font_scale'] * 0.8,
                    text_params['font_thickness'] - 1
                )
                y_pos = text_params['y_start'] + i * text_params['line_spacing']
                x_pos = text_params['x_offset']
                cv2.rectangle(
                    display_frame,
                    (x_pos - text_params['bg_padding'], y_pos - th - text_params['bg_padding']),
                    (x_pos + tw + text_params['bg_padding'], y_pos + text_params['bg_padding']),
                    text_params['bg_color'],
                    -1
                )
                cv2.putText(
                    display_frame, text,
                    (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    text_params['font_scale'] * 0.8,
                    color,
                    text_params['font_thickness'] - 1
                )
        else:
            progress = f"Collecting frames: {len(frame_queue)}/{SEQUENCE_LENGTH}"
            cv2.putText(
                display_frame, progress,
                (text_params['x_offset'], text_params['y_start']),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_params['font_scale'] * 0.8,
                (0, 255, 0),
                text_params['font_thickness'] - 1
            )

        out.write(display_frame)
        cv2.imshow('Video Processing', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    log_file.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = r"path\to\your\video.mp4"
    main(video_path)
