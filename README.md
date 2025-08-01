# Brain-Tumor-Detector-and-Automatic-Report-Generation
Used MobileNetV2 for Classification task.
Used UNet for Segmentation task.
Used T5 transformer + customization report for report generation.
Classification will provide presence and class of tumor, Segmentation will provide the location, area, size of classified tumor.
Result of classification and segmenation will be saved in a json format and passed to T5 finetuned transformer.
Generated data for T5-transformer using above classification and segmentation models.
Used MRI dataset from kaggle for classification and segmentation.
