# ML_project

CMPT726 - SFU

-------------------General--------------

For this project we started reading the paper titled "Frustum PointNets for 3D Object Detection from RGB-D Data" which can be downloaded
from (https://arxiv.org/abs/1711.08488).

We are trying to implement the code which can be found on Github (https://github.com/zhixinwang/frustum-convnet)

The dataset needs to be downloaded and all the links are in ML_project/Docs/dataset-download-links.txt

To know more about the dataset you can check: (https://github.com/yanii/kitti-pcl/blob/master/KITTI_README.TXT)

--------------------GCP-----------------

For training purposes we are planning to use Google Cloud Platform. Please signup for $300 credit if you haven't already on (https://cloud.google.com/free/).
This will require your credit/debit card details so as to verify your identity and charge you if you finish the first $300.

Once your GCP accout is setup, you need to create an instance. You can follow the link https://cloud.google.com/ai-platform/deep-learning-vm/docs/cli
Please be sure to add this part to create an instance for 100 gigs of memory and NVIDIA k80:
gcloud compute instances create "instance-CMPT726" --zone= "us-west1-b" --image-family="pytorch-latest-gpu" --image-project=deeplearning-platform-release --maintenance-policy=TERMINATE --accelerator="type=nvidia-tesla-k80,count=1" --machine-type= "n1-standard-4" --boot-disk-size=100GB --metadata="install-nvidia-driver=True"
PLEASE be sure to STOP or DELETE the created instance so that you are not billed for something you did'nt make use of.


--------------------Local Computer-----------
If you are working on a local host then you might install the required libraries to run the python scripts. We recommend everyone to create an invironment
CMPT726 and always activate this environment where all the libraries are installed.

To install all the libraries you can just run the script created by Thuy which can be downloaded from our working repository ML_project/create_env.sh