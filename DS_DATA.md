# Prepare Downstream Data

First, create directory `./blob/data` and download all the datasets.

### LaPa

* Download LaPa.tar.gz from https://github.com/JDAI-CV/lapa-dataset.
* Uncompress to `./blob/data/LaPa`, make sure `./blob/data/LaPa/{test, train, val}/` all exist.

### CelebAMask-HQ

* Download CelebAMask-HQ.zip from https://github.com/switchablenorms/CelebAMask-HQ.
* Uncompress to `./blob/data/CelebAMask-HQ`, make sure `./blob/data/CelebAMask-HQ/{CelebA-HQ-img, CelebAMask-HQ-mask-anno}/` all exist.

### AFLW-19

* Download the annotations from http://mmlab.ie.cuhk.edu.hk/projects/compositional/AFLWinfo_release.mat to `./blob/data/AFLW19/AFLWinfo_release.mat`.
* Download the images following instructions given by https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/#download. Uncompress the aflw-images-{0,2,3}.tar.gz files to `./blob/data/AFLW19/`, make sure `./blob/data/AFLW19/data/flickr/{0, 2, 3}/` exists.


### IBUG300W & WFLW

* Download the IBUG300W and WFLW annotations from https://github.com/HRNet/HRNet-Facial-Landmark-Detection#data.
* Download IBUG300W images from
    * https://ibug.doc.ic.ac.uk/download/annotations/ibug.zip
    * https://ibug.doc.ic.ac.uk/download/annotations/afw.zip
    * https://ibug.doc.ic.ac.uk/download/annotations/helen.zip
    * https://ibug.doc.ic.ac.uk/download/annotations/lfpw.zip
* Download WFLW images from https://wywu.github.io/projects/LAB/WFLW.html.
* Uncompress these files, make sure these paths exist:
    * IBUG300W images: `./blob/data/IBUG300W/{ibug, afw, helen, lfpw}/`
    * IBUG300W annotations (from HRNet): `./blob/data/IBUG300W/face_landmarks_300w_{train, valid_challenge, valid_common}.csv`
    * WFLW images: `./blob/data/WFLW/WFLW_images/`
    * WFLW annotations (from HRNet): `./blob/data/WFLW/face_landmarks_300w_{train, test, test_{blur, expression, illumination, largepose, makeup, occlusion}}.csv`


The tree of `./blob/data` should look like:

```
blob/data/
│
├── LaPa/
│   ├── test/
│   ├── train/
│   └── val/
│
├── CelebAMask-HQ/
│   ├── CelebA-HQ-img/
│   ├── CelebAMask-HQ-mask-anno/
│   ├── list_eval_partition.txt
│   └── CelebA-HQ-to-CelebA-mapping.txt
│
├── AFLW-19/  
│   ├── AFLWinfo_release.mat
│   └── data/
│       └── flickr/
│ 
├── IBUG300W/
│   ├── ibug/
│   ├── afw/
│   ├── helen/
│   ├── lfpw/
│   ├── face_landmarks_300w_train.csv
│   ├── face_landmarks_300w_valid_challenge.csv
│   └── face_landmarks_300w_valid_common.csv
│
└── WFLW/
    ├── WFLW_images/
    ├── face_landmarks_wflw_test_blur.csv  
    ├── face_landmarks_wflw_test_expression.csv    
    ├── face_landmarks_wflw_test_largepose.csv  
    ├── face_landmarks_wflw_test_occlusion.csv
    ├── face_landmarks_wflw_test.csv       
    ├── face_landmarks_wflw_test_illumination.csv  
    ├── face_landmarks_wflw_test_makeup.csv     
    └── face_landmarks_wflw_train.csv

```

Now let's repack all these datasets into uniform formats for efficient reading. Just run with

```bash
python -m farl.datasets.prepare ./blob/data
```

Finally, we should have the following files under `./blob/data`:

```
LaPa.train.zip
LaPa.test.zip

CelebAMaskHQ.train.zip
CelebAMaskHQ.test.zip

AFLW-19.train.zip
AFLW-19.test.zip
AFLW-19.test_frontal.zip

IBUG300W.train.zip
IBUG300W.test_common.zip
IBUG300W.test_challenging.zip

WFLW.train.zip
WFLW.test_all.zip
WFLW.test_blur.zip
WFLW.test_expression.zip
WFLW.test_illumination.zip
WFLW.test_largepose.zip
WFLW.test_makeup.zip
WFLW.test_occlusion.zip
```
