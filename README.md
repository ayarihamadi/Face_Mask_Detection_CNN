# Face_Mask_Detection_CNN
Face Mask Detection system based on computer vision and deep learning using OpenCV and Pytorch

# Dataset
Download the data set: https://drive.google.com/drive/folders/1-OHsIFdEvLQM7-OHtMecYdqT6aY_Prr1?usp=sharing

# Training
1) Train ResNet50 Model:
  in train_resnet50.ipynb select the data_path (where the dataset is installed) and model_dir (where the trained models should be saved)
  Execute train_resnet50.ipynb
2) Train MobileNetV2 Model:
  in train_mobilenetv2.ipynb select the data_path (where the dataset is installed) and model_dir (where the trained models should be saved)
  Execute train_mobilenetv2.ipynb
  
# Testing
1) Open test.ipynb
1) In the import: select from model_resnet50 import Model to test the ResNet50 model and from model_mobilenetv2 import Model to test the MobileNetV2 model.
2) Select a model_path (The path of the saved model to test), be careful to select the right model for testing either ResNet50 or MobileNetV2.
3) Execute test.ipynb
