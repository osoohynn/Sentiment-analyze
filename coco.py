import cv2
import numpy as np
import trimesh

# Detectron2 관련 라이브러리 임포트
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

# 1. Detectron2를 이용하여 항공 이미지에서 건물 마스크를 추론하는 함수
def get_building_masks(image_path, model_weight_path):
    cfg = get_cfg()
    # COCO 기반 Mask R-CNN 모델 사용 (건물 전용 모델이 있다면 그걸 사용하세요)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # 예: "path/to/your_building_model.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 건물 하나의 클래스로 가정
    cfg.MODEL.DEVICE = "cpu"  # GPU 사용, 없으면 "cpu"
    predictor = DefaultPredictor(cfg)

    image = cv2.imread(image_path)
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    masks = instances.pred_masks.numpy()  # shape: (N, H, W)
    return image, masks

# 2. OpenCV를 이용하여 마스크에서 2D 바운딩 박스(직사각형)를 추출하는 함수
def extract_bounding_boxes(masks):
    rects = []
    for mask in masks:
        # 마스크를 0 또는 255로 변환
        mask_uint8 = (mask.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 가장 큰 외곽선을 선택
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            rects.append((x, y, w, h))
    return rects

# 3. trimesh를 이용해 2D 바운딩 박스를 고정 높이(예: 10미터)로 extrude하여 3D 박스를 생성하는 함수
def create_box_from_rect(rect, height=10.0):
    x, y, w, h = rect
    # 박스의 크기는 [w, h, height] (여기서 w, h는 2D 바운딩 박스의 너비, 높이)
    box = trimesh.creation.box(extents=[w, h, height])
    # 기본 박스의 중심은 (0,0,0)이므로, 박스의 낮은 면(바닥)이 (x, y, 0)이 되도록 이동:
    # 박스 중심을 (w/2, h/2, height/2)로 두었으므로, 전체 이동은 (x + w/2, y + h/2, height/2)
    box.apply_translation([x + w/2, y + h/2, height/2])
    # 모든 정점에 회색 컬러 지정 (RGBA: 128, 128, 128, 255)
    box.visual.vertex_colors = [128, 128, 128, 255]
    return box

# 4. 전체 파이프라인 실행: 건물 마스크 추론 → 바운딩 박스 추출 → 3D 박스 생성 및 OBJ 파일로 저장
def main():
    image_path = "images/pexels-photo-2224931.jpeg"          # 항공 사진 파일 경로
    model_weight_path = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")    # 건물 전용 모델 가중치 경로

    # Detectron2로 건물 마스크 얻기
    image, masks = get_building_masks(image_path, model_weight_path)
    print(f"총 {masks.shape[0]}개의 건물 마스크 추론됨")
    
    # 마스크에서 2D 바운딩 박스 추출
    rects = extract_bounding_boxes(masks)
    print(f"총 {len(rects)}개의 2D 바운딩 박스 추출됨")
    
    # 각 바운딩 박스를 trimesh를 사용해 3D 박스로 변환
    boxes = []
    for rect in rects:
        box = create_box_from_rect(rect, height=10.0)  # 원하는 높이로 조정 가능
        boxes.append(box)
    
    if len(boxes) == 0:
        print("생성된 박스가 없습니다.")
        return
    
    # 여러 박스를 하나의 mesh로 합치기
    combined = trimesh.util.concatenate(boxes)
    
    # 결과를 OBJ 파일로 저장
    combined.export("buildings_boxes.obj")
    print("buildings_boxes.obj 파일로 저장됨")

    # OBJ 파일 불러오기
    mesh = trimesh.load('buildings_boxes.obj')
    # 내장 뷰어로 시각화 (창이 뜨면서 3D 객체를 회전/확대할 수 있습니다)
    mesh.show()

if __name__ == "__main__":
    main()
