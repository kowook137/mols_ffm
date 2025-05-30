#!/usr/bin/env python3
"""
Fisher-Flow MOLS 데이터 전처리 및 캐싱
RTX 3080 듀얼 GPU 환경 최적화
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import wandb
from typing import Dict, List, Tuple

from src.datasets.latin_dataset import LatinSquareDataset, FisherGeometry


def create_splits(data_dir: Path, target_orders: List[int] = [8, 9]) -> Dict[str, Dict]:
    """
    Fisher-Flow 학습용 데이터 split 생성
    
    Split 전략:
    - Train: n ≤ 7 (3,4,5,7)  
    - Val: n = 8
    - Test: n = 9
    """
    splits = {
        'train': {'orders': [3, 4, 5, 7], 'samples': []},
        'val': {'orders': [8], 'samples': []},
        'test': {'orders': [9], 'samples': []}
    }
    
    # 각 split별로 데이터 로드
    for split_name, split_info in splits.items():
        print(f"\n=== {split_name.upper()} Split 생성 ===")
        
        dataset = LatinSquareDataset(
            data_dir=data_dir,
            orders=split_info['orders'],
            transform_to_sphere=True,
            epsilon=1e-4,
            augment_rotations=False,  # 캐싱시에는 augmentation 비활성화
            augment_relabeling=False
        )
        
        split_info['dataset'] = dataset
        split_info['num_samples'] = len(dataset)
        
        print(f"{split_name}: {len(dataset)} samples from orders {split_info['orders']}")
    
    return splits


def cache_tensors(splits: Dict, cache_dir: Path, batch_size: int = 32):
    """
    Fisher-Rao 변환된 텐서들을 .pt 파일로 캐싱
    GPU 메모리 효율적으로 처리
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # GPU 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"GPU 사용: {torch.cuda.get_device_name()}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    for split_name, split_info in splits.items():
        print(f"\n=== {split_name.upper()} 캐싱 중 ===")
        
        dataset = split_info['dataset']
        split_cache = []
        
        # 배치별로 처리 (메모리 효율성)
        for i in tqdm(range(0, len(dataset), batch_size), desc=f"Caching {split_name}"):
            batch_data = []
            
            for j in range(i, min(i + batch_size, len(dataset))):
                sample = dataset[j]
                
                # GPU로 이동 후 처리
                if 'sphere_points' in sample:
                    sample['sphere_points'] = sample['sphere_points'].to(device)
                if 'one_hot' in sample:
                    sample['one_hot'] = sample['one_hot'].to(device)
                if 'square' in sample:
                    sample['square'] = sample['square'].to(device)
                
                batch_data.append(sample)
            
            # CPU로 이동 후 저장
            for sample in batch_data:
                for key in ['sphere_points', 'one_hot', 'square']:
                    if key in sample:
                        sample[key] = sample[key].cpu()
                split_cache.append(sample)
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 캐시 파일 저장
        cache_file = cache_dir / f"{split_name}_cache.pt"
        torch.save({
            'samples': split_cache,
            'orders': split_info['orders'],
            'num_samples': len(split_cache),
            'epsilon': 1e-4,
            'transform_info': 'Fisher-Rao mapping to S^d_+'
        }, cache_file)
        
        print(f"{split_name} 캐시 저장: {cache_file}")
        print(f"파일 크기: {cache_file.stat().st_size / 1e6:.1f} MB")


def validate_cache(cache_dir: Path):
    """캐시된 데이터 검증"""
    print("\n=== 캐시 검증 ===")
    
    geometry = FisherGeometry()
    
    for split in ['train', 'val', 'test']:
        cache_file = cache_dir / f"{split}_cache.pt"
        
        if not cache_file.exists():
            print(f"Warning: {cache_file} not found")
            continue
            
        data = torch.load(cache_file)
        samples = data['samples']
        
        print(f"\n{split.upper()}:")
        print(f"  Samples: {len(samples)}")
        print(f"  Orders: {data['orders']}")
        
        # 첫 번째 샘플 검증
        if samples:
            sample = samples[0]
            
            if 'sphere_points' in sample:
                sphere_pts = sample['sphere_points']
                print(f"  Sphere shape: {sphere_pts.shape}")
                print(f"  Sphere norm: {torch.norm(sphere_pts[0,0]):.4f}")
                
                # Round-trip 테스트
                flat_sphere = sphere_pts.view(-1, sphere_pts.shape[-1])
                recovered = geometry.from_sphere(flat_sphere)
                one_hot_flat = sample['one_hot'].view(-1, sample['one_hot'].shape[-1])
                error = torch.max(torch.abs(recovered - one_hot_flat))
                print(f"  Round-trip error: {error:.6f}")


def create_wandb_artifact(cache_dir: Path, project_name: str = "fisher-flow-mols"):
    """W&B Artifact 생성"""
    print("\n=== W&B Artifact 생성 ===")
    
    # W&B 초기화
    wandb.init(project=project_name, job_type="data-preprocessing")
    
    # Artifact 생성
    artifact = wandb.Artifact(
        name="mols_fisher_cache",
        type="dataset",
        description="Fisher-Flow MOLS dataset with Fisher-Rao transformations",
        metadata={
            "orders": [3, 4, 5, 7, 8, 9],
            "target_orders": [8, 9],
            "epsilon": 1e-4,
            "transform": "Fisher-Rao mapping to S^d_+",
            "split_strategy": "train(≤7), val(8), test(9)"
        }
    )
    
    # 캐시 파일들 추가
    for cache_file in cache_dir.glob("*_cache.pt"):
        artifact.add_file(str(cache_file), name=cache_file.name)
    
    # Artifact 로그
    wandb.log_artifact(artifact)
    wandb.finish()
    
    print("W&B Artifact 저장 완료!")


def get_data_statistics(cache_dir: Path):
    """데이터셋 통계 출력"""
    print("\n=== 데이터셋 통계 ===")
    
    total_samples = 0
    for split in ['train', 'val', 'test']:
        cache_file = cache_dir / f"{split}_cache.pt"
        if cache_file.exists():
            data = torch.load(cache_file)
            samples = len(data['samples'])
            total_samples += samples
            orders = data['orders']
            
            print(f"{split:>5}: {samples:>6} samples (orders: {orders})")
    
    print(f"{'Total':>5}: {total_samples:>6} samples")
    
    # GPU 메모리 요구사항 추정
    if total_samples > 0:
        # 예시: 9x9 square → 9x9x9 one-hot → 729 float32 = ~3KB/sample
        estimated_memory = total_samples * 3 * 1024  # bytes
        print(f"\n추정 GPU 메모리 요구량: {estimated_memory / 1e6:.1f} MB")
        print(f"RTX 3080 (10GB) 배치 크기 권장: 32-64")


def main():
    parser = argparse.ArgumentParser(description="Fisher-Flow MOLS 데이터 전처리")
    parser.add_argument("--data-dir", type=str, default="data/raw", 
                       help="원본 MOLS 데이터 디렉토리")
    parser.add_argument("--cache-dir", type=str, default="data/cache",
                       help="캐시 저장 디렉토리") 
    parser.add_argument("--batch-size", type=int, default=32,
                       help="캐싱 배치 크기")
    parser.add_argument("--wandb", action="store_true",
                       help="W&B Artifact 생성")
    parser.add_argument("--validate-only", action="store_true",
                       help="캐시 검증만 실행")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    
    if args.validate_only:
        validate_cache(cache_dir)
        get_data_statistics(cache_dir)
        return
    
    print("🔄 Fisher-Flow MOLS 데이터 전처리 시작")
    print(f"원본 데이터: {data_dir}")
    print(f"캐시 저장: {cache_dir}")
    
    # 1. 데이터 split 생성
    splits = create_splits(data_dir)
    
    # 2. 텐서 캐싱
    cache_tensors(splits, cache_dir, args.batch_size)
    
    # 3. 캐시 검증
    validate_cache(cache_dir)
    
    # 4. 통계 출력
    get_data_statistics(cache_dir)
    
    # 5. W&B Artifact (선택사항)
    if args.wandb:
        create_wandb_artifact(cache_dir)
    
    print("\n✅ 데이터 전처리 완료!")
    print(f"다음 단계: python cli.py train --data-cache {cache_dir}")


if __name__ == "__main__":
    main() 