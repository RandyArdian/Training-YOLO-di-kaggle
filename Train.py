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

model = YOLO("YOLO/yolo11n_SPDConv.yaml")
results = model.train(data="VisDrone.yaml",epochs=100,imgsz=640,resume=True)
