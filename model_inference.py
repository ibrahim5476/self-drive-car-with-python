from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("valll_data", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "valll_data")
print(inference_on_dataset(predictor.model, val_loader, evaluator))