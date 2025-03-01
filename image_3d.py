import os
import numpy as np
import cv2
import torch
import trimesh
from PIL import Image
from torchvision import transforms

# 필요한 라이브러리가 설치되어 있는지 확인
required_packages = {
    "torch": "pip install torch torchvision",
    "PIL": "pip install Pillow",
    "trimesh": "pip install trimesh",
    "numpy": "pip install numpy",
    "cv2": "pip install opencv-python"
}

missing_packages = []
for package, install_cmd in required_packages.items():
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(install_cmd)

if missing_packages:
    print("다음 패키지를 설치해야 합니다:")
    for cmd in missing_packages:
        print(cmd)
    print("설치 후 다시 시도하세요.")
    # exit()

# HEIC 파일 지원
try:
    from pillow_heif import register_heif_opener
    has_heif_support = True
    register_heif_opener()
except ImportError:
    has_heif_support = False
    print("HEIC 파일을 지원하려면: pip install pillow_heif")

# 3D-R2N2 모델 클래스 (간소화된 버전)
class SimpleEncoder(torch.nn.Module):
    def __init__(self):
        super(SimpleEncoder, self).__init__()
        # 이미지를 인코딩하는 CNN 레이어
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        # 특징을 3D 복셀 그리드로 변환하는 FC 레이어
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(256 * 7 * 7, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 32 * 32 * 32)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_layers(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = x.view(batch_size, 32, 32, 32)
        return x

# 간단한 3D-R2N2 모델 클래스
class Simple3DR2N2:
    def __init__(self, pretrained_weights=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"사용 중인 디바이스: {self.device}")
        
        self.model = SimpleEncoder().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 사전 훈련된 가중치 로드 (실제 구현에서는 가중치 파일 필요)
        if pretrained_weights and os.path.exists(pretrained_weights):
            self.model.load_state_dict(torch.load(pretrained_weights, map_location=self.device))
            print(f"가중치 로드됨: {pretrained_weights}")
        else:
            print("사전 훈련된 가중치를 찾을 수 없습니다. 더미 추론을 실행합니다.")
        
        self.model.eval()
    
    def predict(self, image_path):
        """단일 이미지로부터 3D 복셀 그리드 예측"""
        # 이미지 로드 및 전처리
        try:
            if image_path.lower().endswith(('.heic', '.heif')) and has_heif_support:
                img = Image.open(image_path)
                # HEIC를 일시적으로 JPG로 변환
                temp_jpg = "temp_converted.jpg"
                img.save(temp_jpg)
                img = Image.open(temp_jpg)
            else:
                img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"이미지 로드 오류: {e}")
            return None
        
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # 추론
        with torch.no_grad():
            voxel_grid = self.model(img_tensor)
            
        # 임계값 적용하여 이진 복셀 그리드 생성
        voxel_grid = (voxel_grid > 0.5).float()
        return voxel_grid.cpu().numpy()[0]

# 복셀 그리드를 메시로 변환
def voxel_grid_to_mesh(voxel_grid, threshold=0.5):
    """복셀 그리드를 3D 메시로 변환"""
    from skimage import measure
    
    # Marching Cubes 알고리즘 적용
    try:
        verts, faces, normals, values = measure.marching_cubes(voxel_grid, threshold)
        
        # 메시 생성
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, normals=normals)
        return mesh
    except Exception as e:
        print(f"메시 생성 오류: {e}")
        return None

# 메인 함수
def image_to_3d_model_dl(image_path, output_path="output_model.obj", simplify=True):
    """딥러닝을 사용하여 이미지에서 3D 모델 생성"""
    
    # 1. 모델 초기화
    model = Simple3DR2N2()
    
    # 2. 이미지에서 복셀 그리드 예측
    print(f"이미지 처리 중: {image_path}")
    voxel_grid = model.predict(image_path)
    
    if voxel_grid is None:
        print("복셀 그리드 생성 실패")
        return None
    
    # 3. 복셀 그리드에서 메시 생성
    print("복셀 그리드에서 메시 생성 중...")
    mesh = voxel_grid_to_mesh(voxel_grid)
    
    if mesh is None:
        print("메시 생성 실패")
        return None
    
    # 4. 메시 단순화 (선택 사항)
    if simplify and len(mesh.faces) > 1000:
        print("메시 단순화 중...")
        mesh = mesh.simplify_quadratic_decimation(1000)
    
    # 5. 메시 저장
    mesh.export(output_path)
    print(f"3D 모델 저장됨: {output_path}")
    
    # 6. 시각화 (선택 사항)
    try:
        mesh.show()
    except:
        print("메시 시각화 실패 - 모델은 저장되었습니다.")
    
    return mesh

# 더미 데이터로 단순 복셀 모델 생성 (실제 AI 모델이 없을 때 사용)
def generate_dummy_3d_model(image_path, output_path="dummy_model.obj", shape_type="cube"):
    """
    실제 딥러닝 모델 없이 이미지 분석을 통해 간단한 3D 모델 생성
    """
    # 이미지 로드
    try:
        if image_path.lower().endswith(('.heic', '.heif')) and has_heif_support:
            from PIL import Image
            img = np.array(Image.open(image_path))
        else:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"이미지 로드 오류: {e}")
        return None
        
    # 이미지 분석
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 가장 큰 윤곽선 찾기
    if not contours:
        print("윤곽선을 찾을 수 없습니다.")
        return None
        
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 형태 분석
    approx = cv2.approxPolyDP(largest_contour, 0.02 * cv2.arcLength(largest_contour, True), True)
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w) / h
    
    # 형태 결정
    if shape_type == "auto":
        if len(approx) == 4 and 0.9 <= aspect_ratio <= 1.1:
            shape_type = "cube"  # 정사각형 -> 정육면체
        elif len(approx) == 4:
            shape_type = "box"   # 직사각형 -> 직육면체
        elif len(approx) == 3:
            shape_type = "pyramid"  # 삼각형 -> 피라미드
        elif cv2.isContourConvex(approx):
            shape_type = "sphere"  # 볼록 형태 -> 구
        else:
            shape_type = "cylinder"  # 기타 -> 원기둥
    
    # 3D 모델 생성
    if shape_type == "cube":
        # 정육면체
        size = max(w, h) / 100
        mesh = trimesh.primitives.Box(extents=[size, size, size])
    elif shape_type == "box":
        # 직육면체
        mesh = trimesh.primitives.Box(extents=[w/100, h/100, min(w, h)/150])
    elif shape_type == "pyramid":
        # 피라미드
        mesh = trimesh.creation.cone(radius=min(w, h)/200, height=max(w, h)/100)
    elif shape_type == "sphere":
        # 구
        mesh = trimesh.primitives.Sphere(radius=min(w, h)/200)
    elif shape_type == "cylinder":
        # 원기둥
        mesh = trimesh.primitives.Cylinder(radius=min(w, h)/200, height=max(w, h)/100)
    else:
        # 기본값: 직육면체
        mesh = trimesh.primitives.Box(extents=[w/100, min(w, h)/100, h/100])
    
    # 복셀화 - 뭉툭한 느낌을 내기 위해
    voxel_size = 0.05
    voxelized = mesh.voxelized(voxel_size)
    mesh = voxelized.as_boxes()
    
    # 모델 저장
    mesh.export(output_path)
    print(f"간단한 3D 모델 저장됨: {output_path}")
    
    # 시각화
    try:
        mesh.show()
    except:
        print("메시 시각화 실패 - 모델은 저장되었습니다.")
    
    return mesh

if __name__ == "__main__":
    # 입력 이미지 경로
    image_path = "images/pyramid.jpg"  # 사용자의 실제 이미지 경로로 변경하세요
    
    # 모델 가중치 파일 경로 (없으면 더미 생성)
    weights_path = "3d_r2n2_weights.pth"  # 실제 가중치 파일 경로
    
    if os.path.exists(weights_path):
        # 실제 딥러닝 모델 사용
        image_to_3d_model_dl(image_path, "output_model_dl.obj")
    else:
        print("사전 훈련된 가중치를 찾을 수 없습니다. 더미 모델을 생성합니다.")
        print("다양한 형태 생성:")
        
        # 다양한 형태로 더미 모델 생성하여 테스트
        shape_types1 = ["cube", "box", "pyramid", "sphere", "cylinder"]
        shape_types = ["pyramid", "sphere", "cylinder"]
        for shape in shape_types:
            output_path = f"output_model_{shape}.obj"
            generate_dummy_3d_model(image_path, output_path, shape)