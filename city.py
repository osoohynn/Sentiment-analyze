import numpy as np
import cv2
import trimesh
from scipy import ndimage
from skimage import measure

def image_to_simple_3d(image_path, save_path="simple_model.obj"):
    """
    이미지를 간단한 3D 모델로 변환합니다.
    건물은 직육면체로 단순화됩니다.
    """
    # 이미지 로드 및 전처리
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 에지 감지를 통한 건물 윤곽 찾기
    edges = cv2.Canny(gray, 100, 200)
    
    # 윤곽선 찾기
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 빈 장면 생성
    scene = trimesh.Scene()
    
    # 각 윤곽선을 직육면체로 변환
    for contour in contours:
        # 면적이 너무 작은 윤곽선 무시
        if cv2.contourArea(contour) < 500:
            continue
        
        # 윤곽선에 바운딩 박스 적용
        x, y, w, h = cv2.boundingRect(contour)
        
        # 높이 추정 (간단히 너비의 1.5배로 설정)
        height = w * 1.5
        
        # 직육면체 생성
        box = trimesh.primitives.Box(
            extents=[w/50, height/50, h/50],
            transform=trimesh.transformations.translation_matrix(
                [(x + w/2)/50 - img.shape[1]/100, height/100, (y + h/2)/50 - img.shape[0]/100]
            )
        )
        
        # 장면에 추가
        scene.add_geometry(box)
    
    # 모델 저장
    scene.export(save_path)
    print(f"3D 모델이 {save_path}에 저장되었습니다.")
    
    # 시각화 (선택 사항)
    scene.show()
    
    return scene

def depth_map_to_3d(image_path, save_path="depth_model.obj"):
    """
    이미지의 명암을 깊이 맵으로 사용하여 단순한 3D 지형 생성
    """
    # 이미지 로드 및 전처리
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 가우시안 블러 적용하여 부드럽게
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 깊이 매핑 (밝은 부분이 높게)
    depth_map = img_blur.astype(float) / 255.0
    
    # 메시 생성 (Marching Cubes 알고리즘)
    verts, faces, _, _ = measure.marching_cubes(depth_map, 0.5)
    
    # Trimesh 메시로 변환
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    
    # 단순화하여 뭉툭하게
    mesh = mesh.simplify_quadratic_decimation(len(mesh.faces) // 3)
    
    # 모델 저장
    mesh.export(save_path)
    print(f"3D 모델이 {save_path}에 저장되었습니다.")
    
    # 시각화 (선택 사항)
    mesh.show()
    
    return mesh

def dl_image_to_blocks(image_path, save_path="block_model.obj"):
    """
    이미지에서 단순한 블록 형태의 3D 모델 생성
    """
    # 이미지 로드
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 이미지 세그먼테이션 (간단히 임계값으로)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # 레이블링
    labeled, num_labels = ndimage.label(thresh)
    
    # 장면 생성
    scene = trimesh.Scene()
    
    for i in range(1, num_labels + 1):
        # 레이블 마스크
        mask = (labeled == i).astype(np.uint8) * 255
        
        # 블록 위치 및 크기 계산
        y, x = ndimage.center_of_mass(mask)
        area = np.sum(mask) / 255
        size = np.sqrt(area) / 10
        
        # 높이는 면적에 비례하도록
        height = size * 2
        
        # 블록 생성
        block = trimesh.primitives.Box(
            extents=[size, height, size],
            transform=trimesh.transformations.translation_matrix(
                [x/50 - img.shape[1]/100, height/2, y/50 - img.shape[0]/100]
            )
        )
        
        # 장면에 추가
        scene.add_geometry(block)
    
    # 모델 저장
    scene.export(save_path)
    print(f"3D 모델이 {save_path}에 저장되었습니다.")
    
    # 시각화 (선택 사항)
    scene.show()
    
    return scene

def image_to_shape_3d(image_path, save_path="shape_model.obj"):
    # 이미지 로드 및 전처리
    img = load_image(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # 윤곽선 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    scene = trimesh.Scene()
    
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        
        # 형태 추정
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 꼭지점 수에 따라 다른 3D 형태 생성
        corners = len(approx)
        x, y, w, h = cv2.boundingRect(contour)
        
        if corners == 4:  # 사각형 - 육면체
            # 정사각형에 가까운지 확인
            aspect_ratio = float(w) / h
            
            if 0.9 <= aspect_ratio <= 1.1:  # 정사각형에 가까움
                size = max(w, h) / 50
                box = trimesh.primitives.Box(
                    extents=[size, size, size],  # 정육면체
                    transform=trimesh.transformations.translation_matrix(
                        [(x + w/2)/50 - img.shape[1]/100, size/2, (y + h/2)/50 - img.shape[0]/100]
                    )
                )
            else:  # 직사각형
                box = trimesh.primitives.Box(
                    extents=[w/50, h/50, min(w, h)/50],  # 직육면체
                    transform=trimesh.transformations.translation_matrix(
                        [(x + w/2)/50 - img.shape[1]/100, h/100, (y + h/2)/50 - img.shape[0]/100]
                    )
                )
            scene.add_geometry(box)
            
        elif corners == 3:  # 삼각형 - 피라미드
            # 삼각형의 중심 계산
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # 피라미드 생성
            height = max(w, h) / 50
            vertices = []
            for point in approx:
                px, py = point[0]
                vertices.append([(px - img.shape[1]/2)/50, 0, (py - img.shape[0]/2)/50])
            
            # 꼭대기 점 추가
            vertices.append([(cx - img.shape[1]/2)/50, height, (cy - img.shape[0]/2)/50])
            
            # 면 생성
            faces = []
            for i in range(len(approx)):
                faces.append([i, (i+1) % len(approx), len(approx)])
            
            # 바닥면 추가
            bottom_face = list(range(len(approx)-1, -1, -1))
            faces.append(bottom_face)
            
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            scene.add_geometry(mesh)
            
        elif corners == 5:  # 오각형 - 원통
            cylinder = trimesh.primitives.Cylinder(
                radius=min(w, h)/100,
                height=max(w, h)/50,
                transform=trimesh.transformations.translation_matrix(
                    [(x + w/2)/50 - img.shape[1]/100, max(w, h)/100, (y + h/2)/50 - img.shape[0]/100]
                )
            )
            scene.add_geometry(cylinder)
            
        else:  # 기타 형태 - 원기둥 또는 구
            if cv2.isContourConvex(approx):  # 볼록 형태는 구로 표현
                sphere = trimesh.primitives.Sphere(
                    radius=min(w, h)/100,
                    transform=trimesh.transformations.translation_matrix(
                        [(x + w/2)/50 - img.shape[1]/100, min(w, h)/100, (y + h/2)/50 - img.shape[0]/100]
                    )
                )
                scene.add_geometry(sphere)
            else:  # 비볼록 형태는 직육면체로 단순화
                box = trimesh.primitives.Box(
                    extents=[w/50, min(w, h)/50, h/50],
                    transform=trimesh.transformations.translation_matrix(
                        [(x + w/2)/50 - img.shape[1]/100, min(w, h)/100, (y + h/2)/50 - img.shape[0]/100]
                    )
                )
                scene.add_geometry(box)
    
    # 모델 저장
    scene.export(save_path)
    print(f"3D 모델이 {save_path}에 저장되었습니다.")
    
    # 시각화
    scene.show()
    
    return scene

# image_to_simple_3d("images/jung.png")
# depth_map_to_3d("building.jpg")
# dl_image_to_blocks("images/jung.png")
image_to_shape_3d("images/jung.png")