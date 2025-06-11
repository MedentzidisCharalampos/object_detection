def test_model_runs():
    from backend.detection.yolov5 import YOLOv5Detector
    model = YOLOv5Detector("yolov5s.pt")
    class_names, confs, boxes = model.detect_and_save("data/sample.jpg", "data/sample_out.jpg")
    assert len(class_names) > 0
