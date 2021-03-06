## SSNet Detection on Wideband signal

## Overview

This project implemented SSNet for Wideband signal detection and tested on actual received wideband signal. 

### Example Output

![Bounding box detection result of a example image.](./figures/bounding_box.jpeg "Bounding box detection result of an image in the KITTI dataset.")

> *Bounding box detection result of an time frequency spectrum of broadband signal

![](./figures/center_heatmap.jpeg)

> *Predicted heatmap of object center points on an time frequency spectrum of broadband signal

## File Structure

```
├── SSNet
│   ├── dataset.py
│   ├── DLAnet.py
│   ├── loss.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── dataset_split
│   ├── train.txt
│   └── val.txt
├── environment.yml
```

This repository was developed and tested in PyTorch 1.5.

## How to run

- Intall required dependencies as listed in [environment.yml](./environment.yml)
- Modify signal dataset directory in centernet-vanilla/dataset.py
- Run [train.py](SSNet/train.py) for training and [predict.py](SSNet/predict.py) for inference


## Results

![](./figures/results.png)

> *Compare evaluation results of our implementation to the original CenterNet on KITTI.*


![](./figures/compare.jpeg)

> *An example image in the validation set. (left) Ground truth (right) inference results from our implementation.*

![](./figures/heatmap_small.jpeg)

> *(left) Ground truth (right) Predicted heatmap. (bottom) Inference results.*

Figure above shows an example inference result compared to the ground truth. It is shown that our model to able to predict most of the objects correctly in this scene. The comparison between the ground truth heatmap with Gaussian smoothing and our predicted heatmap on the same image is also shown on the image above.

![](./figures/pr_curve.png)

> *Precision Recall curve on validation set.*

Figure above shows the precision-recall curve of our final model on the validation set. Three curves represent easy, moderate, and hard objects respectively. The area under the curve is the average precision (AP).

## Acknowledgement

We used the DLA-34 network, loss functions and some other functions from this [R-CenterNet repo](https://github.com/ZeroE04/R-CenterNet).
