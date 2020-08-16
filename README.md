# CGAN-filter
By [Or Tslil](https://github.com/ortslil64) and Avishy Carmi
A recursive state estimator based on conditional GANs (CGAN), performs better than the optimal estimators (Stein phenomena)
## Sequential images filtering
It can be used to estimate sequance of images given anothere sequance. In the next figure, we demonstrate our algorithm deals with unobservables observations.
The images are sampled from a sequance with skips of 3 images per sample. The observation image is partially blocked by a tree. The rows are arranged from top down: observations,ground-truth , CGAN filter, pix2pix estimator

![demo](https://github.com/ortslil64/CGAN-filter/blob/master/images/illsutration.png?raw=true "Under the tree the object are not observable")

## State estimation compared with Kalman filter and particle filter
Anothere application is to filter process, given an observation with a hidden model. The time-seies is preprocessed as a tensor an normalized.
The following figure shows the CGAN filter performance (mean square error (MSE) and bias) compared with the Kalman filter, for filtering a linear process.
The top figure shows in red is the MSE of the Kalman filter, in blue is the MSE of the CGAN filter, in  black the Cramer–Rao bound and in dashed blackthe new tighter bound.
The bottom figure shows the bias of the CGAN filter over training. The both figures are 200 Monte-Carloruns of 450 training epochs.
Each epoch consists of 327680 samples and is divided to 40 128×128 images who overlapeach other.
![demo](https://github.com/ortslil64/CGAN-filter/blob/master/images/MSE.png?raw=true "Under the tree the object are not observable")

## dataset
The dataset is taken from https://github.com/ortslil64/SPO-dataset.

![demo](https://github.com/ortslil64/SPO-dataset/blob/master/images/partal_example_tree.png?raw=true "Under the tree the object are not observable")


## Pipelines
The models pipelines are visualized in the next figure. each component (predictor, corrector and likelihood) are based on pix2pix architecture with Unet as its ganerative model.
![demo](https://github.com/ortslil64/CGAN-filter/blob/master/images/model.png?raw=true "Pipelines")
## Installation
1) Clone the repository:
```bash
git clone https://github.com/ortslil64/CGAN-filter.git
```

2) Install the dependencies in `requirments.txt` file

3) Install the package using pip:
```bash
pip install -e CGAN-filter
```

4) (optional) Download, unzip the demo model weights and save them into `demo/video filtering/model_weights`:

```bash
wget https://googledrive.com/
```

## Usage

### Time-series filtering
Run the demo file:
```bash
python demo/process\ filtering/train.py  
```
The file can be modified for different filtering tasks (linear\ non-linear\ multiple inputs\ etc.)

### Video filtering
Run the demo file:
```bash
python demo/video\ filtering/train.py  
```

## References
Not yet published, "Better-than-optimal filtering with conditional GANs"