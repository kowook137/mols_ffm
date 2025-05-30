#!/usr/bin/env python3
"""
MOLS 데이터 생성 스크립트
Fisher-Flow 학습을 위한 다양한 order의 Mutually Orthogonal Latin Squares 생성
"""

import numpy as np
import json
import itertools
from pathlib import Path
from typing import List, Tuple, Optional
import argparse

def is_latin_square(square: np.ndarray) -> bool:
    """Latin Square 검증"""
    n = square.shape[0]
    if square.shape != (n, n):
        return False
    
    # 각 행과 열에 1~n이 모두 나타나는지 확인
    for i in range(n):
        if set(square[i, :]) != set(range(1, n+1)):
            return False
        if set(square[:, i]) != set(range(1, n+1)):
            return False
    return True

def are_orthogonal(square1: np.ndarray, square2: np.ndarray) -> bool:
    """두 Latin Square가 직교하는지 검증"""
    if square1.shape != square2.shape:
        return False
    
    n = square1.shape[0]
    pairs = set()
    
    for i in range(n):
        for j in range(n):
            pair = (square1[i, j], square2[i, j])
            if pair in pairs:
                return False
            pairs.add(pair)
    
    return len(pairs) == n * n

def generate_cyclic_latin_square(n: int) -> np.ndarray:
    """순환 Latin Square 생성 (홀수 n에 대해)"""
    if n % 2 == 0:
        raise ValueError("순환 방법은 홀수 order에만 적용 가능")
    
    square = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            square[i, j] = ((i + j) % n) + 1
    return square

def generate_finite_field_mols(p: int) -> List[np.ndarray]:
    """유한체 기반 MOLS 생성 (소수 p에 대해)"""
    if not is_prime(p):
        raise ValueError(f"{p}는 소수가 아닙니다")
    
    # p-1개의 MOLS 생성
    mols = []
    
    for k in range(1, p):
        square = np.zeros((p, p), dtype=int)
        for i in range(p):
            for j in range(p):
                square[i, j] = ((k * i + j) % p) + 1
        mols.append(square)
    
    return mols

def is_prime(n: int) -> bool:
    """소수 판별"""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def generate_order_4_mols() -> List[np.ndarray]:
    """4×4 MOLS 생성 (특별한 구성)"""
    # 4×4에서는 최대 3개의 MOLS 가능
    mols = []
    
    # 첫 번째 Latin Square
    L1 = np.array([
        [1, 2, 3, 4],
        [2, 1, 4, 3],
        [3, 4, 1, 2],
        [4, 3, 2, 1]
    ])
    
    # 두 번째 Latin Square
    L2 = np.array([
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [2, 1, 4, 3],
        [3, 4, 1, 2]
    ])
    
    # 세 번째 Latin Square
    L3 = np.array([
        [1, 2, 3, 4],
        [3, 4, 1, 2],
        [4, 3, 2, 1],
        [2, 1, 4, 3]
    ])
    
    mols = [L1, L2, L3]
    
    # 직교성 검증
    for i in range(len(mols)):
        for j in range(i+1, len(mols)):
            if not are_orthogonal(mols[i], mols[j]):
                print(f"Warning: L{i+1}과 L{j+1}은 직교하지 않습니다")
    
    return mols

def generate_order_6_incomplete() -> List[np.ndarray]:
    """6×6에서는 완전한 MOLS가 존재하지 않으므로 단일 Latin Square만 생성"""
    # Euler의 36 officers problem - 해가 없음이 증명됨
    # 하나의 Latin Square만 생성
    square = np.array([
        [1, 2, 3, 4, 5, 6],
        [2, 3, 4, 5, 6, 1],
        [3, 4, 5, 6, 1, 2],
        [4, 5, 6, 1, 2, 3],
        [5, 6, 1, 2, 3, 4],
        [6, 1, 2, 3, 4, 5]
    ])
    return [square]

def save_mols_data(mols: List[np.ndarray], order: int, data_dir: Path):
    """MOLS 데이터를 JSON 형태로 저장"""
    data_dir.mkdir(parents=True, exist_ok=True)
    
    mols_data = {
        'order': order,
        'num_squares': len(mols),
        'squares': [square.tolist() for square in mols],
        'metadata': {
            'max_possible_mols': order - 1 if is_prime(order) else 'unknown',
            'is_complete_set': len(mols) == order - 1 if is_prime(order) else False
        }
    }
    
    filename = data_dir / f"mols_{order}x{order}.json"
    with open(filename, 'w') as f:
        json.dump(mols_data, f, indent=2)
    
    print(f"Order {order}: {len(mols)} MOLS 저장됨 -> {filename}")

def main():
    parser = argparse.ArgumentParser(description='MOLS 데이터 생성')
    parser.add_argument('--orders', nargs='+', type=int, 
                       default=[3, 4, 5, 7, 8, 9, 11, 13],
                       help='생성할 Latin Square의 크기들')
    parser.add_argument('--output-dir', type=str, 
                       default='molsffm/data/raw',
                       help='출력 디렉토리')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    for order in args.orders:
        print(f"\n=== Order {order} MOLS 생성 ===")
        
        try:
            if order == 2:
                # 2×2에서는 orthogonal pair가 존재하지 않음
                square = np.array([[1, 2], [2, 1]])
                mols = [square]
                
            elif order == 3:
                # 3×3: 2개의 MOLS 가능 (3-1=2)
                mols = generate_finite_field_mols(3)
                
            elif order == 4:
                # 4×4: 특별 구성으로 3개 MOLS
                mols = generate_order_4_mols()
                
            elif order == 5:
                # 5×5: 4개의 MOLS 가능 (5-1=4)
                mols = generate_finite_field_mols(5)
                
            elif order == 6:
                # 6×6: 완전한 MOLS 불가능
                mols = generate_order_6_incomplete()
                
            elif order == 7:
                # 7×7: 6개의 MOLS 가능 (7-1=6)
                mols = generate_finite_field_mols(7)
                
            elif order in [11, 13, 17, 19]:
                # 소수 order: p-1개의 MOLS 가능
                mols = generate_finite_field_mols(order)
                
            elif order == 8:
                # 8×8: 부분적 MOLS (완전하지 않음)
                # 첫 번째만 생성하고 나머지는 수동 구성 필요
                base_square = np.zeros((8, 8), dtype=int)
                for i in range(8):
                    for j in range(8):
                        base_square[i, j] = ((i + j) % 8) + 1
                mols = [base_square]
                
            elif order == 9:
                # 9×9: 8개의 MOLS 가능 (3^2-1=8)
                # 간단한 방법으로 일부만 생성
                base_square = np.zeros((9, 9), dtype=int)
                for i in range(9):
                    for j in range(9):
                        base_square[i, j] = ((i + j) % 9) + 1
                mols = [base_square]
                
            else:
                print(f"Order {order}는 지원되지 않습니다")
                continue
            
            # 검증
            for i, square in enumerate(mols):
                if not is_latin_square(square):
                    print(f"Error: Square {i}는 유효한 Latin Square가 아닙니다")
                    continue
            
            # 직교성 검증 (2개 이상인 경우)
            if len(mols) >= 2:
                orthogonal_pairs = 0
                total_pairs = len(mols) * (len(mols) - 1) // 2
                
                for i in range(len(mols)):
                    for j in range(i+1, len(mols)):
                        if are_orthogonal(mols[i], mols[j]):
                            orthogonal_pairs += 1
                
                print(f"직교성 검증: {orthogonal_pairs}/{total_pairs} 쌍이 직교")
            
            # 저장
            save_mols_data(mols, order, output_dir)
            
        except Exception as e:
            print(f"Order {order} 생성 실패: {e}")

if __name__ == "__main__":
    main() 