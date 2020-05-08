# ObjectDetectionCreator
This software package is for creating object detection models.
Please reach out to me if you are interested in creating a custom object detection program I would be glad to help!

1. Define your object classes in labelingApp/windows_v1.2/data/predefined_classes.txt and in data/labelmap.pbtxt

2. Add images to the Images/... testImages, trainImages, and validationImages
    Atleast each object should be represented 20 times

3. Configure faster_rcnn_inception_v2_coco.config
    Lines: 
    10 num_classes
    Change the first part of the filepaths at the lines listed below to fit your system
    107 fine_tune_checkpoint 
    119 input_path
    121 label_map_path
    133 input_path  
    135 label_map_path

4. Use train.py to create a model, change the filepaths as detailed at the top of the script

5. Use export_model.py to build the trained model, change the filepaths as detailed at the top of the script

6. Use processGenerativeImage.py to build a larger training set, change the filepaths as detailed at the top of the script

Based on the tutorial:
https://www.geeksforgeeks.org/ml-training-image-classifier-using-tensorflow-object-detection-api/
