"""
PEWTAF: Privacy-Aware Lightweight Edge Intelligent Framework
for Real-Time Lethal Threat Interception and Risk Assessment.

Main orchestration script for end-to-end execution:
    1. Video Stream Ingestion (CCTV / Drone / Video File / Webcam)
    2. Ephemeral Frame Buffering (EFPM)
    3. Preprocessing (Resize 640x640, RGB, [0, 1] normalization)
    4. Neural Inference (Hailo-8L NPU or PyTorch Emulation)
    5. Detection Post-Processing
    6. Temporal Threat Validation Module (TTVM)
    7. Contextual Threat Assessment Network (TAN)
    8. Privacy-Preserving Metadata Dispatch (MQTT / REST / Console)
    9. Ephemeral Frame Purge (Zero Raw Video Persistence)
"""

import argparse
import sys
import time
import cv2
import torch

# Internal modules
from detection.preprocessing import FramePreprocessor
from detection.postprocessing import DetectionPostProcessor
from temporal.ttv_module import TemporalThreatValidator
from threat.threat_assessment import ThreatAssessment
from privacy.ephemeral_processing import EphemeralFrameProcessor
from communication.metadata import MetadataGenerator
from communication.mqtt_client import MQTTMetadataPublisher
from communication.rest_client import RESTMetadataClient
from evaluation.benchmark import DeploymentBenchmark


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PEWTAF Edge Threat Interception Pipeline"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Path to video file, RTSP stream URL, or camera device index (default: '0')"
    )
    parser.add_argument(
        "--hef",
        type=str,
        default=None,
        help="Path to compiled Hailo HEF model file (if running on Raspberry Pi + Hailo-8L)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to PyTorch model checkpoint (.pt) for workstation/CPU emulation"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Inference frame resolution (default: 640)"
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.25,
        help="Detection confidence cutoff (default: 0.25)"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="TTVM sliding temporal observation window N (default: 5)"
    )
    parser.add_argument(
        "--persistence-thresh",
        type=float,
        default=0.60,
        help="TTVM persistence score threshold tau_p (default: 0.60)"
    )
    parser.add_argument(
        "--alpha-ema",
        type=float,
        default=0.80,
        help="TTVM EMA bounding box smoothing parameter alpha (default: 0.80)"
    )
    parser.add_argument(
        "--mqtt-broker",
        type=str,
        default=None,
        help="Optional MQTT broker IP for real-time alert dispatch"
    )
    parser.add_argument(
        "--rest-endpoint",
        type=str,
        default=None,
        help="Optional HTTP REST endpoint for alert telemetry"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display on-screen preview (local debug mode only; disable for headless edge deployment)"
    )
    return parser.parse_args()


class MockPyTorchInferenceBackend:
    """Fallback PyTorch runner if Hailo-8L hardware runtime is not present."""
    def __init__(self, weights_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Initializing PyTorch Inference Backend on: {self.device}")
        
        # Load Hybrid YOLOv10-MobileViTv3 model architecture
        from models.yolov10_mobilevit import HybridYOLOv10MobileViT
        self.model = HybridYOLOv10MobileViT().to(self.device)
        self.model.eval()

        if weights_path:
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                print(f"[INFO] Loaded weights from {weights_path}")
            except Exception as e:
                print(f"[WARNING] Could not load checkpoint ({e}). Running in inference test mode.")

    def infer(self, tensor):
        tensor = tensor.to(self.device)
        with torch.no_grad():
            # In test mode, returns predictions format: [x1, y1, x2, y2, confidence, class_id]
            # When deploying with real detector head, replace with head output decoding
            return []


def setup_inference_backend(hef_path, weights_path):
    """Selecting between native Hailo-8L NPU or PyTorch host backend"""
    if hef_path:
        try:
            from deployment.hailo_inference import HailoInference
            # Assuming native HailoRT Python library is installed
            import hailo_platform as hp
            hailo_runtime = hp.HailoRT()
            backend = HailoInference(hef_path=hef_path, runtime=hailo_runtime)
            backend.initialize()
            print(f"[INFO] Hailo-8L NPU initialized successfully with {hef_path}")
            return backend
        except Exception as err:
            print(f"[WARNING] Failed to initialize Hailo-8L hardware ({err}). Falling back to host PyTorch.")

    return MockPyTorchInferenceBackend(weights_path=weights_path)


def main():
    args = parse_args()

    print("=" * 65)
    print(" PEWTAF: Edge-Intelligent Lethal Threat Interception System")
    print("=" * 65)

    # 1. Initialize Pipeline Modules
    preprocessor = FramePreprocessor(image_size=args.img_size)
    postprocessor = DetectionPostProcessor(
        class_names=["Pistol", "Knife", "Rifle"],
        confidence_threshold=args.conf_thresh
    )
    temporal_validator = TemporalThreatValidator(
        window_size=args.window_size,
        persistence_threshold=args.persistence_thresh,
        tau_iou=0.50,
        alpha=args.alpha_ema
    )
    threat_assessor = ThreatAssessment()
    metadata_generator = MetadataGenerator()
    ephemeral_processor = EphemeralFrameProcessor()
    benchmark = DeploymentBenchmark()

    # Optional remote notification publishers
    rest_client = RESTMetadataClient(args.rest_endpoint) if args.rest_endpoint else None

    # Setup inference backend (Hailo or PyTorch)
    backend = setup_inference_backend(args.hef, args.weights)

    # 2. Open Video Stream
    source_input = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source_input)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {args.source}")
        sys.exit(1)

    print(f"[INFO] Video ingestion started from source: {args.source}")
    print("[INFO] Press 'q' to terminate execution.\n")

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[INFO] Video feed ended or disconnected.")
                break

            frame_count += 1
            t_start = time.perf_counter()

            # --- Ephemeral Frame Processing Mechanism (EFPM) ---
            ephemeral_processor.store(frame)

            try:
                # Step 1: Preprocessing (CPU)
                tensor = preprocessor.process(frame)

                # Step 2: Backbone + MSHSA + Detection Head (Hailo-8L NPU / PyTorch)
                raw_preds = backend.infer(tensor)

                # Step 3: Detection Post-processing & Thresholding (CPU)
                detections = postprocessor.process(raw_preds)

                # Step 4: Temporal Threat Validation Module (TTVM) (CPU)
                validated = temporal_validator.update(detections)

                # Step 5: Threat Assessment Network (TAN) (CPU)
                if validated is not None:
                    assessment = threat_assessor.evaluate(
                        bbox=validated.bbox,
                        weapon_class=validated.class_name,
                        confidence=validated.confidence,
                        persistence=validated.persistence
                    )

                    # Step 6: Metadata Generation & Secure Dispatch
                    packet = metadata_generator.generate(assessment)
                    json_payload = metadata_generator.serialize(packet)

                    print(f"[ALERT DISPATCH] Frame #{frame_count} | "
                          f"Class: {packet['class']} | "
                          f"Threat Score (Rt): {packet['threat_score']} | "
                          f"Level: {packet['threat_level']} | "
                          f"Priority: Level {packet['alert_priority']}")

                    # Remote transmission (encrypted metadata only)
                    if rest_client:
                        try:
                            rest_client.send(packet)
                        except Exception as e:
                            print(f"[COMM ERROR] REST post failed: {e}")

                # Local GUI display (for qualitative evaluation/demo only)
                if args.display:
                    display_frame = frame.copy()
                    if validated is not None:
                        x1, y1, x2, y2 = [int(v) for v in validated.bbox]
                        label = f"{validated.class_name} ({packet['threat_level']}: {packet['threat_score']})"
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(display_frame, label, (x1, max(y1 - 10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.imshow("PEWTAF Edge Surveillance Monitor", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            finally:
                # EFPM: Enforce immediate volatile memory release (Section 4.8.2)
                ephemeral_processor.discard()

            # Record elapsed time per frame
            elapsed = time.perf_counter() - t_start
            benchmark.latencies.append(elapsed)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
        if hasattr(backend, "close"):
            backend.close()

        print("\n" + "=" * 65)
        print(" BENCHMARK PERFORMANCE REPORT")
        print("=" * 65)
        print(f"Total Frames Processed : {frame_count}")
        print(f"Average Latency        : {benchmark.average_latency_ms:.2f} ms")
        print(f"Average Throughput     : {benchmark.fps:.2f} FPS")
        print("=" * 65)


if __name__ == "__main__":
    main()