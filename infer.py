import argparse

import cv2
import supervision as sv

from detector import Detector


def parse_args():
    parser = argparse.ArgumentParser(description="RF-DETR-Seg Video Inference")
    parser.add_argument("--source", type=str, default=0)
    parser.add_argument("--model", type=str, default="small",
                        choices=["small", "medium", "large", "xlarge", "2xlarge"])
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    detector = Detector(model_size=args.model, threshold=args.threshold)

    video_capture = cv2.VideoCapture(args.source)
    if not video_capture.isOpened():
        raise RuntimeError(f"Failed to open video source: {args.source}")

    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator()

    while True:
        success, frame_bgr = video_capture.read()
        if not success:
            break

        detections = detector.detect(frame_bgr)
        labels = Detector.labels(detections)

        annotated = mask_annotator.annotate(frame_bgr, detections)
        annotated = label_annotator.annotate(annotated, detections, labels)

        cv2.imshow("RF-DETR-Seg", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
