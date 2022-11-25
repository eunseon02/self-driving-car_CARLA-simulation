# self-driving-car_CARLA-simulation

## 1) Carla simulator 환경 구축 
운영체제는 Ubuntu 20.04를 이용했다.
clang-8 설치
github의 Carla simulator에서 CARLA 0.9.13 package 압축파일 다운로드 및 압축해제 후 실행해준다. 

## 2) ROS1 설치 및 환경 구축




## 3) YOLOP 적용
( YOLOP는 객체 감지(Object detection) 분야에서 많이 알려진 모델 중 하나로 객체 감지(traffic object detection), 주행 가능 영역(drivable area segmentation), 차선 인식(lane detection)이 동시에 가능하다. )

### -환경 구축
-CUDA설치
-Pytorch 설치

YOLOP를 gitclone 해왔다. 문제는 코드가 demo파일에서 어떠한 경로로부터 이미지나 동여상 파일을 불러와서 이미지 분석을 마친 뒤 다시 코드 상의 output path에 저장시킨다는 점이였는데 YOLOP의 코드 자체가 여러 파일로 아주 복잡하게 구성되어 있어 시간 관계상 demo.py 파일에서 output 이미지를 찾아 이를 자율구행을 구현한 test4.py 파일에 불러오는 방식을 이용하였다. 이렇게 처리된 이미지는 다음과 같다.
![simu](https://user-images.githubusercontent.com/108911413/203894258-d7313532-0a0f-445e-a449-e0207762a46a.jpg)
이 이미지를 binary image로 변환 후 opencv의 bird eye view 관련 함수를 이용해 다음과 같은 이미지로 변환해준다.
![제목 없음](https://user-images.githubusercontent.com/108911413/203894156-3c759610-a1b4-4995-bda3-17247829cd17.png)

결과는 다음과 같다. 커브 구간에서 차선을 밟는다는 점이 아쉬웠는데 전시회 때 다른 자율주행 시뮬레이터를 써보셨던 선배님께서 차선을 차의 바로 앞부분만 인식해서 처리하지 말고 먼거리까지 처리해서 곡선도로의 시작과 끝을 찍고 그 사이에 적합한 원의 지름을 구하여 path-planning 하는 방식을 이용해보라고 하셨다..

https://user-images.githubusercontent.com/108911413/203893261-eeaab50f-d703-4eeb-b646-fad97cc81394.mp4

