# Background
To extract shorelines from Sentinel-1 data, land and water can be classified through histogram analysis and automatic thresholding (see DE Africa’s existing notebook). However, the thresholding method can be affected by the proportion of land/water present on the image. More importantly, it was observed that some coastal lands, e.g., sandy flat beaches were misclassified due to its low backscattering values like water. Therefore, it is expected that a supervised machine learning model would be potentially helpful to more stable land/water classification and less misclassification of coastal lands.  

# What does this repository conatin
This repository contains codes and notebooks of a workflow for coastline mapping using Sentinel-1 data and supervised image classification. It contains three notebooks:  
* 0_Generate_training_data_multiple_locations.ipynb: This notebook classifies coastal land and water using Sentinel-2 data using MNDWI, identifies class labels from the classified Sentinel-2 data, pre-processes Sentinel-1 data and collect training features from Sentinel-1 data. In the notebook five experiemnt locations are selected.  
* 1_Train_assess_model_by_location.ipynb: This notebook builds a supervised classification model from the collected training data for each experiement location, assesses models and export the trained models.  
* 2_Classify_extract_shorelines.ipynb: This notebook applies the trained models to classify coastal land and water, implements tidal modelling and filtering, and extracts annual shorelines from Sentinel-1 data. It also compares classification results between Sentinel-2 and Sentinel-1.  
* 3_Threshold_extract_shorelines.ipynb: This notebook compares automatic thresholding based on histogram and supervised thresholding using samples from Sentinel-2 data and ROC for shoreline mapping.

# How to use the repository
Run the three notebooks in sequence to implement the workflow. The notebooks are intended to run in DE Africa's Sandbox environment.  

# Additional notes
This repository may be updated frequently. A report on the workflow and findings is being produced and published.
