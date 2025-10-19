# Generative-Visual-Attribute-Guided-Embeddings

This repository contains two fully scripts that implement a **a real-time person re-identification system**.  
The design integrates **vision-language reasoning**, **deep re-identification**, and **semantic attribute matching** to track individuals across frames and detect prolonged presence in a scene.


## Features

- **YOLOv8 Detection:** Person detection on input video streams.  
- **OSNet-based Re-ID:** Deep appearance embeddings using TorchReID’s feature extractor.  
- **Moondream Attribute Extraction:** Vision-language model describes clothing and attributes from cropped person images.  
- **Semantic Attribute Matching:** Combines fuzzy word similarity, sentence-transformer embeddings, and synonym expansion for robust matching.  
- **Color Histogram Matching:** Adds texture/color consistency across frames.  
- **Hungarian Assignment:** Associates detections with tracks using a weighted multi-factor similarity matrix.  
- **Loitering Detection:** Flags individuals remaining in view beyond a configurable threshold.  
- **Edge-Optimized:** Designed for Jetson AGX / Orin environments with GPU acceleration.


