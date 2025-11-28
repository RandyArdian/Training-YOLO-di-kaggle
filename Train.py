!git clone https://github.com/RandyArdian/Riset.git
%cd Riset
!pip install -e .

import ultralytics
from ultralytics import YOLO
ultralytics.checks()

!pip install protobuf==3.20.*
!pip install tensorboard==2.14
!pip install numpy==1.26.4


#title Select YOLO11 🚀 logger {run: 'auto'}
logger = 'TensorBoard' #@param ['TensorBoard', 'Weights & Biases']

if logger == 'TensorBoard':
  !yolo settings tensorboard=True
  %load_ext tensorboard
  %tensorboard --logdir .
elif logger == 'Weights & Biases':
  !yolo settings wandb=True
  
#Train YOLO11n
!yolo train model=yolo11n.pt data=coco8.yaml epochs=3 imgsz=640

# Validate YOLO11n (contoh)
!yolo val model=yolo11n.pt data=coco8.yaml
