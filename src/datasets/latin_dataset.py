#!/usr/bin/env python3
"""
Latin Square Dataset for Fisher-Flow
Fisher-Rao manifold 변환을 포함한 MOLS 데이터셋 구현
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from torch.utils.data import Dataset, DataLoader

class FisherGeometry:
    """Fisher-Rao geometry on the positive orthant of hypersphere"""
    
    @staticmethod
    def to_sphere(one_hot: torch.Tensor, epsilon: float = 1e-4) -> torch.Tensor:
        """
        Fisher-Flow의 핵심 변환: Δ^d → S^d_+
        
        Args:
            one_hot: Categorical distribution or one-hot encoded tensor [..., d]
            epsilon: Label smoothing parameter for numerical stability
            
        Returns:
            Points on positive orthant of d-hypersphere [..., d]
        """
        # Label smoothing: p̃ = (1-ε)p + ε/d
        batch_shape = one_hot.shape[:-1]
        d = one_hot.shape[-1]
        
        # Label smoothing for numerical stability
        uniform = torch.ones_like(one_hot) / d
        smoothed = (1 - epsilon) * one_hot + epsilon * uniform
        
        # Fisher-Rao mapping: φ(p) = √p (element-wise square root)
        sphere_point = torch.sqrt(smoothed)
        
        # Ensure we're on unit sphere (should be automatic, but numerical safety)
        sphere_point = F.normalize(sphere_point, p=2, dim=-1)
        
        return sphere_point
    
    @staticmethod
    def from_sphere(sphere_point: torch.Tensor) -> torch.Tensor:
        """
        Inverse transformation: S^d_+ → Δ^d
        
        Args:
            sphere_point: Points on positive orthant of d-hypersphere [..., d]
            
        Returns:
            Probability distributions [..., d]
        """
        # Inverse Fisher-Rao mapping: φ^{-1}(s) = s^2
        prob = sphere_point ** 2
        
        # Normalize to ensure it's a valid probability distribution
        prob = prob / prob.sum(dim=-1, keepdim=True)
        
        return prob
    
    @staticmethod
    def geodesic_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Fisher-Rao geodesic distance on S^d_+
        
        Args:
            x, y: Points on positive orthant of d-hypersphere [..., d]
            
        Returns:
            Geodesic distances [...]
        """
        # Clamp to avoid numerical issues with arccos
        dot_product = torch.sum(x * y, dim=-1)
        dot_product = torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)
        
        return torch.arccos(dot_product)
    
    @staticmethod
    def geodesic_interpolation(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Geodesic interpolation on S^d_+ (closed-form solution)
        
        Args:
            x0: Starting points [..., d]
            x1: End points [..., d] 
            t: Interpolation parameter [..., 1] or scalar
            
        Returns:
            Interpolated points on geodesics [..., d]
        """
        # Ensure t has correct shape
        if t.dim() == 0:  # scalar
            t = t.expand(x0.shape[:-1] + (1,))
        elif t.shape[-1] != 1:
            t = t.unsqueeze(-1)
        
        # Geodesic distance
        dot_product = torch.sum(x0 * x1, dim=-1, keepdim=True)
        dot_product = torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.arccos(dot_product)
        
        # Handle case where points are identical (theta ≈ 0)
        sin_theta = torch.sin(theta)
        small_theta = (theta < 1e-6)
        
        # For small theta, use linear interpolation to avoid division by zero
        alpha = torch.where(small_theta, 1 - t, torch.sin((1 - t) * theta) / sin_theta)
        beta = torch.where(small_theta, t, torch.sin(t * theta) / sin_theta)
        
        result = alpha * x0 + beta * x1
        
        # Normalize to ensure we stay on the sphere
        result = F.normalize(result, p=2, dim=-1)
        
        return result

class LatinSquareDataset(Dataset):
    """
    MOLS 데이터셋 - Fisher-Flow용으로 구성
    """
    
    def __init__(
        self, 
        data_dir: Union[str, Path],
        orders: List[int] = [3, 4, 5, 7],
        transform_to_sphere: bool = True,
        epsilon: float = 1e-4,
        augment_rotations: bool = True,
        augment_relabeling: bool = True,
        max_samples_per_order: Optional[int] = None
    ):
        """
        Args:
            data_dir: MOLS 데이터가 저장된 디렉토리
            orders: 사용할 Latin Square 크기들
            transform_to_sphere: Fisher-Rao 변환 적용 여부
            epsilon: Label smoothing parameter
            augment_rotations: 회전/반사 augmentation
            augment_relabeling: 심볼 재라벨링 augmentation
            max_samples_per_order: 각 order당 최대 샘플 수
        """
        self.data_dir = Path(data_dir)
        self.orders = orders
        self.transform_to_sphere = transform_to_sphere
        self.epsilon = epsilon
        self.augment_rotations = augment_rotations
        self.augment_relabeling = augment_relabeling
        self.max_samples_per_order = max_samples_per_order
        
        # Training mode 초기화 (기본값: False)
        self.training = False
        
        self.samples = []
        self.geometry = FisherGeometry()
        
        self._load_data()
    
    def _load_data(self):
        """MOLS 데이터 로드 및 전처리"""
        for order in self.orders:
            filepath = self.data_dir / f"mols_{order}x{order}.json"
            
            if not filepath.exists():
                print(f"Warning: {filepath} not found, skipping order {order}")
                continue
                
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            squares = data['squares']
            num_squares = len(squares)
            
            print(f"Loading order {order}: {num_squares} squares")
            
            # 각 Latin Square를 개별 샘플로 추가
            for i, square in enumerate(squares):
                square_tensor = torch.tensor(square, dtype=torch.long) - 1  # 0-indexed
                
                sample = {
                    'square': square_tensor,
                    'order': order,
                    'square_id': i,
                    'num_squares': num_squares
                }
                
                self.samples.append(sample)
                
                # 최대 샘플 수 제한
                if (self.max_samples_per_order is not None and 
                    len([s for s in self.samples if s['order'] == order]) >= self.max_samples_per_order):
                    break
            
            # MOLS 쌍 생성 (직교성 학습용)
            if num_squares >= 2:
                for i in range(min(num_squares, 3)):  # 최대 3개까지
                    for j in range(i + 1, min(num_squares, 3)):
                        pair_sample = {
                            'square_pair': [
                                torch.tensor(squares[i], dtype=torch.long) - 1,
                                torch.tensor(squares[j], dtype=torch.long) - 1
                            ],
                            'order': order,
                            'pair_id': (i, j),
                            'is_orthogonal': True  # MOLS에서 가져온 것이므로 직교
                        }
                        self.samples.append(pair_sample)
        
        print(f"Total samples loaded: {len(self.samples)}")
    
    def _to_one_hot(self, square: torch.Tensor) -> torch.Tensor:
        """Latin Square를 one-hot encoding으로 변환"""
        order = square.shape[0]
        one_hot = F.one_hot(square, num_classes=order).float()  # [order, order, order]
        return one_hot
    
    def _augment_square(self, square: torch.Tensor) -> torch.Tensor:
        """Latin Square augmentation (회전, 반사, 재라벨링)"""
        if not (self.augment_rotations or self.augment_relabeling):
            return square
        
        augmented = square.clone()
        
        # 회전/반사 augmentation
        if self.augment_rotations and torch.rand(1) < 0.5:
            # 90도 회전
            if torch.rand(1) < 0.25:
                augmented = torch.rot90(augmented, k=1, dims=(0, 1))
            # 180도 회전  
            elif torch.rand(1) < 0.5:
                augmented = torch.rot90(augmented, k=2, dims=(0, 1))
            # 270도 회전
            elif torch.rand(1) < 0.75:
                augmented = torch.rot90(augmented, k=3, dims=(0, 1))
            # 수평 반사
            elif torch.rand(1) < 0.875:
                augmented = torch.flip(augmented, dims=[0])
            # 수직 반사
            else:
                augmented = torch.flip(augmented, dims=[1])
        
        # 심볼 재라벨링
        if self.augment_relabeling and torch.rand(1) < 0.3:
            order = square.shape[0]
            perm = torch.randperm(order)
            augmented = perm[augmented]
        
        return augmented
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """데이터셋 아이템 반환"""
        sample = self.samples[idx]
        
        if 'square' in sample:
            # 단일 Latin Square
            square = sample['square']
            
            # Augmentation 적용
            if self.training:
                square = self._augment_square(square)
            
            # One-hot encoding
            one_hot = self._to_one_hot(square)  # [order, order, order]
            
            result = {
                'square': square,
                'one_hot': one_hot,
                'order': sample['order'],
                'square_id': sample['square_id']
            }
            
            # Fisher-Rao 변환
            if self.transform_to_sphere:
                # Flatten spatial dimensions for transformation
                batch_shape = one_hot.shape[:-1]  # [order, order]
                flat_one_hot = one_hot.view(-1, one_hot.shape[-1])  # [order*order, order]
                
                sphere_points = self.geometry.to_sphere(flat_one_hot, self.epsilon)
                sphere_points = sphere_points.view(batch_shape + (-1,))  # [order, order, order]
                
                result['sphere_points'] = sphere_points
            
            return result
            
        elif 'square_pair' in sample:
            # MOLS 쌍
            square1, square2 = sample['square_pair']
            
            if self.training:
                square1 = self._augment_square(square1)
                square2 = self._augment_square(square2)
            
            one_hot1 = self._to_one_hot(square1)
            one_hot2 = self._to_one_hot(square2)
            
            result = {
                'square_pair': [square1, square2],
                'one_hot_pair': [one_hot1, one_hot2],
                'order': sample['order'],
                'pair_id': sample['pair_id'],
                'is_orthogonal': sample['is_orthogonal']
            }
            
            if self.transform_to_sphere:
                sphere1 = self.geometry.to_sphere(
                    one_hot1.view(-1, one_hot1.shape[-1]), self.epsilon
                ).view(one_hot1.shape)
                
                sphere2 = self.geometry.to_sphere(
                    one_hot2.view(-1, one_hot2.shape[-1]), self.epsilon
                ).view(one_hot2.shape)
                
                result['sphere_pair'] = [sphere1, sphere2]
            
            return result
    
    def train(self):
        """Training mode 설정"""
        self.training = True
        return self
    
    def eval(self):
        """Evaluation mode 설정"""
        self.training = False
        return self


def create_dataloader(
    data_dir: Union[str, Path],
    orders: List[int] = [3, 4, 5, 7],
    batch_size: int = 32,
    shuffle: bool = True,
    transform_to_sphere: bool = True,
    epsilon: float = 1e-4,
    **kwargs
) -> DataLoader:
    """
    Fisher-Flow용 DataLoader 생성
    
    Args:
        data_dir: 데이터 디렉토리
        orders: 사용할 Latin Square 크기들
        batch_size: 배치 크기
        shuffle: 데이터 셔플 여부
        transform_to_sphere: Fisher-Rao 변환 적용
        epsilon: Label smoothing parameter
        **kwargs: Dataset 추가 인자들
        
    Returns:
        DataLoader 인스턴스
    """
    dataset = LatinSquareDataset(
        data_dir=data_dir,
        orders=orders,
        transform_to_sphere=transform_to_sphere,
        epsilon=epsilon,
        **kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )


# 테스트 및 예제 함수들
def test_fisher_geometry():
    """Fisher-Rao geometry 변환 테스트"""
    geometry = FisherGeometry()
    
    # 테스트 데이터 생성 (3x3 Latin Square)
    square = torch.tensor([
        [0, 1, 2],
        [1, 2, 0], 
        [2, 0, 1]
    ])
    
    # One-hot encoding
    one_hot = F.one_hot(square, num_classes=3).float()
    print(f"Original one-hot shape: {one_hot.shape}")
    print(f"One-hot sample:\n{one_hot[0, 0]}")
    
    # Fisher-Rao 변환
    flat_one_hot = one_hot.view(-1, 3)
    sphere_points = geometry.to_sphere(flat_one_hot)
    sphere_points = sphere_points.view(3, 3, 3)
    
    print(f"Sphere points shape: {sphere_points.shape}")
    print(f"Sphere point sample: {sphere_points[0, 0]}")
    print(f"Sphere point norm: {torch.norm(sphere_points[0, 0])}")
    
    # 역변환 테스트
    recovered = geometry.from_sphere(sphere_points.view(-1, 3)).view(3, 3, 3)
    print(f"Recovered shape: {recovered.shape}")
    print(f"Recovery error: {torch.max(torch.abs(recovered - one_hot))}")
    
    # Geodesic interpolation 테스트
    x0 = sphere_points[0, 0]
    x1 = sphere_points[1, 1] 
    t = torch.tensor(0.5)
    
    interpolated = geometry.geodesic_interpolation(x0, x1, t)
    print(f"Geodesic interpolation: {interpolated}")
    print(f"Interpolated norm: {torch.norm(interpolated)}")


if __name__ == "__main__":
    test_fisher_geometry() 