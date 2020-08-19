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

<p align="center">
  <img src="https://github.com/ortslil64/CGAN-filter/blob/master/images/MSE.png" width="500" alt="accessibility text">
</p>

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

https://drive.google.com/file/d/1iTlpMKYUM2k2bEPxOgtkuoL3-mmnujfd/view?usp=sharing


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

## API tutorial

### cganfilter.models.video_filter
This is a filter to estimate sequance of images given their corresponding observations.
Methods for DeepNoisyBayesianFilter(hist = 4, image_shape = (128,128)):
`hist` is the number of images to take as condition (Markov Model of order hist).
`image_shape` is the shape (2d) for the output and input image.
1) train_predictor(x_old, x_new)
Train the predictor CGAN for input x_old and output x_new. x_old shape must be [image_shape[0],image_shape[1], hist] and x_new shape must be  [image_shape[0],image_shape[1]].

2) train_likelihood(z_new, x_new)
Trains the likelihood CGAN. z_new is the observations of x_new (hidden model). z_new has the same shape as x_new.

3) train_updator(x_old, x_new, z_new)
Train the corrector CGAN.

4) propogate(x_old)
Predicts the next state given the last one (x_old), forecasting.

5) estimate(z_new)
Using the likelihood (inverse mmeasurment CGAN) component the estimate the state given the measurment z_new.

6) predict_mean(x_old,z_new)
Estimates the current state, given the last estimated states x_old and observation z_new. This should be recursive, so that the next estimated state is the first in the x_old array of states (with length 'hist').

7) save_weights(path)
Saves the weights of the 3 CGANs models in the path directory.

7) load_weights(path)
Loads the weights of the 3 CGANs models from the path directory.



Example to train the predictor:
```python
from cganfilter.models.video_filter i(mport DeepNoisyBayesianFilter

# ---- Parameters ---- #
hist = 4     
img_shape = (128,128) 
epochs = 100
min_img = 20

# ---- Get the dataset ---- #
x_train, z_train = get_dataset_from_somewhere()
n_train = len(x_train)
 
df = DeepNoisyBayesianFilter(hist,img_shape)

for epoch in tqdm(range(epochs)):
  for t in range(min_img+hist,n_train-1):
      x_new = x_train[t].copy()
      x_old = x_train[t-hist:t].copy()
      df.train_predictor(x_old, x_new)

```

### cganfilter.models.time_serias_filter
This is a filter to estimate sequance of images given their corresponding observations.


Example to train the predictor:
```python
from cganfilter.models.time_serias_filter import DeepNoisyBayesianFilter

# ---- Parameters ---- #
plot_flag = True
measure_gap = 100
img_shape = (64,64)
min_img = 20
skips = img_shape[0]*img_shape[1]//2
n_crop = 100
min_training =  2000
mc_runs = 100


# ---- Get the dataset ---- #
x, z = get_dataset_from_somewhere()
n_train = len(x_train)
 
df = DeepNoisyBayesianFilter(sh = img_shape[0], H = 0.5, loss_type = "l1") 

for t in range(min_img*img_shape[0]*img_shape[1], len(x), img_shape[0]*img_shape[1]//2):
  # ----- Train on the current history ---- #
  for itr in range(t-min_img*img_shape[0]*img_shape[1], t-2*img_shape[0]*img_shape[1], skips):
      ii = itr + np.random.randint(-skips//2, skips//2)
      if ii < 0: ii = 0
      z_old = z[ii:(ii+img_shape[0]*img_shape[1])]
      x_old = x[ii:(ii+img_shape[0]*img_shape[1])]
      z_new = z[(ii+img_shape[0]*img_shape[1]):(ii+2*img_shape[0]*img_shape[1])]
      x_new = x[(ii+img_shape[0]*img_shape[1]):(ii+2*img_shape[0]*img_shape[1])]
      if training_ep < min_training:
        df.train_predictor(x_old, x_new)
        df.train_noise_predictor(z_new)
      if  training_ep > min_training:
        df.train_updator(x_old,x_new,z_new)    
      training_ep += 1

  # ----- Use model to smooth ---- #
  if training_ep > min_training:
      z_new = z[(itr+img_shape[0]*img_shape[1]):(itr+2*img_shape[0]*img_shape[1])]  
      x_new = x[(itr+img_shape[0]*img_shape[1]):(itr+2*img_shape[0]*img_shape[1])] 
      idxs_new = idxs[(itr+img_shape[0]*img_shape[1]):(itr+2*img_shape[0]*img_shape[1])]
      if u is not None:
          u_new = u[(itr+img_shape[0]*img_shape[1]):(itr+2*img_shape[0]*img_shape[1])]
      else:
          u_new = None
      x_old = x[(itr):(itr+img_shape[0]*img_shape[1])]             
      x_hat_df = df.predict_mean(x_old, z_new)
           

```

