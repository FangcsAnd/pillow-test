#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
枕头智能推荐系统 Pillow Recommender V2
========================================
单文件版本：包含预训练模型 + 训练模块 + 推理接口

输入：身高(cm)、体重(kg)、肩宽(cm)、颈椎曲度(21维数组)、性别
输出：推荐枕头参数(头高/颈高/侧高/硬度) + 产品库最佳匹配

用法：
    直接推荐（默认预训练模型）：
        python pillow_recommender_v2.py --height 175 --weight 70 --shoulder 17 \
            --neck-curve '[0,0,0,0,0.8,1,2.7,3.3,3.5,3.4,2.3,1.4,1.4,1.1,0.3,0.1,0,0,0,0,0]' \
            --gender 男

    重新训练（用新数据更新模型）：
        python pillow_recommender_v2.py --train \
            --users users.json --records records.json --pillows pillows.json

依赖：numpy
"""

import json
import numpy as np
import os
import argparse

# ============================================================
# 预训练模型参数 (Pillow Model v1.0) - 硬编码，无需额外文件
# ============================================================

PARAM_MODEL = {
    "head_h": {
        "beta": [111.8033, -29.8232, -12.9790, 12.4045, 1.7173, -11.3916, 1.7173, -4.3036, 12.3682, 1.5679, 9.1483, -2.8334, 2.1337, 3.3464],
        "r2": 0.9934
    },
    "neck_h": {
        "beta": [113.4426, -15.0476, -6.6559, 6.3113, 0.9959, -5.5096, 0.9959, -2.0117, 6.7428, 0.4361, 4.4078, -1.1825, 1.4954, 1.5254],
        "r2": 0.9862
    },
    "side_h": {
        "beta": [111.8033, -29.8232, -12.9790, 12.4045, 1.7173, -11.3916, 1.7173, -4.3036, 12.3682, 1.5679, 9.1483, -2.8334, 2.1337, 3.3464],
        "r2": 0.9934
    },
    "hardness": {
        "beta": [17.7049, 4.6598, 1.8889, -1.8700, -0.1005, 2.0888, -0.1005, 0.8796, -1.2809, -0.7054, -1.6993, 0.7633, 0.2099, -0.7344],
        "r2": 0.9898
    },
    "X_mean": [170.8333, 61.9167, 15.9500, 1.0286, 3.1000, 21.6000, 1.1788, 8.3333, 7.0000, 0.8667, 2.1452, 0.0738, 0.6667],
    "X_std": [6.3355, 10.1834, 1.5861, 0.2836, 0.7616, 5.9554, 0.2933, 1.2472, 1.1547, 0.3273, 0.5964, 0.0735, 0.4714]
}

COMFORT_MODEL = {
    "beta": [2.0242, 1.2122, 0.3982, -1.0795, -0.2005, 0.8351, -0.2005, 0.2297, -0.5969, -0.5990, -0.6900, 0.0591, 0.1524, -0.7068, -2.4329, 2.4721, -0.1867, 0.4778, -0.4431],
    "X_mean": [169.8049, 58.1707, 15.5220, 1.1230, 3.2073, 23.5829, 1.2318, 8.6829, 7.2195, 0.8923, 2.3679, 0.1087, 0.6098, 101.6220, 105.5488, 110.7439, 11.6341, 17.0488],
    "X_std": [7.2321, 11.0125, 1.4910, 0.2770, 0.6564, 5.8167, 0.2620, 1.2185, 1.2975, 0.3070, 0.5696, 0.0762, 0.4878, 41.9290, 27.4844, 37.1985, 7.5555, 9.4983],
    "r2": 0.3095
}

PILLOWS = [
    {"id": "p001", "brand": "亚朵鹅绒枕", "material": "羽绒", "head_height": 187, "neck_height": 155, "side_height": 187, "center_hardness": 10, "neck_hardness": 6},
    {"id": "p002", "brand": "翼眠TPE枕", "material": "其他", "head_height": 89, "neck_height": 103, "side_height": 89, "center_hardness": 22, "neck_hardness": 36},
    {"id": "p003", "brand": "慕思枕", "material": "羽绒", "head_height": 67, "neck_height": 82, "side_height": 93, "center_hardness": 0, "neck_hardness": 12},
    {"id": "p004", "brand": "亚朵3.0", "material": "记忆棉", "head_height": 114, "neck_height": 114, "side_height": 114, "center_hardness": 16, "neck_hardness": 14},
    {"id": "p005", "brand": "水晶家纺硅胶枕", "material": "其他", "head_height": 91, "neck_height": 101, "side_height": 101, "center_hardness": 3, "neck_hardness": 10},
    {"id": "p006", "brand": "张御草本枕", "material": "荞麦", "head_height": 80, "neck_height": 80, "side_height": 80, "center_hardness": 18, "neck_hardness": 18},
    {"id": "p007", "brand": "拉芙菲尔", "material": "记忆棉", "head_height": 98, "neck_height": 114, "side_height": 106, "center_hardness": 17, "neck_hardness": 26},
    {"id": "p008", "brand": "未知型号", "material": "记忆棉", "head_height": 88, "neck_height": 120, "side_height": 88, "center_hardness": 19, "neck_hardness": 15},
    {"id": "p009", "brand": "未知型号", "material": "记忆棉", "head_height": 63, "neck_height": 74, "side_height": 110, "center_hardness": 11, "neck_hardness": 14},
    {"id": "p010", "brand": "百度枕", "material": "记忆棉", "head_height": 75, "neck_height": 86, "side_height": 100, "center_hardness": 3, "neck_hardness": 17},
    {"id": "p011", "brand": "未知型号", "material": "记忆棉", "head_height": 68, "neck_height": 90, "side_height": 109, "center_hardness": 11, "neck_hardness": 20},
    {"id": "p012", "brand": "未知型号", "material": "记忆棉", "head_height": 50, "neck_height": 50, "side_height": 50, "center_hardness": 14, "neck_hardness": 14}
]


def extract_neck_features(neck_curve):
    curve = np.array(neck_curve, dtype=float)
    max_val = np.max(curve)
    return {
        'neck_curve_mean': float(np.mean(curve)),
        'neck_curve_max': float(max_val),
        'neck_curve_area': float(np.sum(curve)),
        'neck_curve_std': float(np.std(curve)),
        'neck_curve_peak_idx': int(np.argmax(curve)),
        'neck_curve_width': int(len([c for c in curve if c > 0.5 * max_val])),
        'neck_curve_front': float(np.mean(curve[:7])),
        'neck_curve_mid': float(np.mean(curve[7:14])),
        'neck_curve_back': float(np.mean(curve[14:])),
    }


def predict_params(height, weight, shoulder_width, neck_curve, gender):
    nf = extract_neck_features(neck_curve)
    features = np.array([
        height, weight, shoulder_width,
        nf['neck_curve_mean'], nf['neck_curve_max'], nf['neck_curve_area'],
        nf['neck_curve_std'], nf['neck_curve_peak_idx'], nf['neck_curve_width'],
        nf['neck_curve_front'], nf['neck_curve_mid'], nf['neck_curve_back'],
        1 if gender == '男' else 0
    ], dtype=float)
    X_mean = np.array(PARAM_MODEL['X_mean'])
    X_std = np.array(PARAM_MODEL['X_std'])
    features_norm = (features - X_mean) / X_std
    features_b = np.concatenate([[1.0], features_norm])
    params = {}
    for name in ['head_h', 'neck_h', 'side_h', 'hardness']:
        beta = np.array(PARAM_MODEL[name]['beta'])
        pred = float(features_b @ beta)
        params[name] = max(0, pred)
    return params


def predict_comfort(height, weight, shoulder_width, neck_curve, gender, pillow):
    nf = extract_neck_features(neck_curve)
    head_h = float(pillow.get('head_height', 0))
    neck_h = float(pillow.get('neck_height', 0))
    side_h = float(pillow.get('side_height', 0))
    center_h = float(pillow.get('center_hardness', 0))
    neck_hard = float(pillow.get('neck_hardness', 0))
    features = np.array([
        height, weight, shoulder_width,
        nf['neck_curve_mean'], nf['neck_curve_max'], nf['neck_curve_area'],
        nf['neck_curve_std'], nf['neck_curve_peak_idx'], nf['neck_curve_width'],
        nf['neck_curve_front'], nf['neck_curve_mid'], nf['neck_curve_back'],
        1 if gender == '男' else 0,
        head_h, neck_h, side_h, center_h, neck_hard
    ], dtype=float)
    X_mean = np.array(COMFORT_MODEL['X_mean'])
    X_std = np.array(COMFORT_MODEL['X_std'])
    features_norm = (features - X_mean) / X_std
    features_b = np.concatenate([[1.0], features_norm])
    beta = np.array(COMFORT_MODEL['beta'])
    return float(features_b @ beta)


def find_best_pillow(height, weight, shoulder_width, neck_curve, gender):
    scored = [(predict_comfort(height, weight, shoulder_width, neck_curve, gender, p), p) 
              for p in PILLOWS]
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored


def recommend(height, weight, shoulder_width, neck_curve, gender='未知'):
    params = predict_params(height, weight, shoulder_width, neck_curve, gender)
    scored = find_best_pillow(height, weight, shoulder_width, neck_curve, gender)
    best = scored[0]
    return {
        'recommended_params': {
            '后脑勺高度_mm': int(params['head_h']),
            '颈椎高度_mm': int(params['neck_h']),
            '侧睡区高度_mm': int(params['side_h']),
            '软硬度': round(params['hardness'], 1)
        },
        'recommended_pillow': {
            'id': best[1]['id'], 'brand': best[1]['brand'],
            'material': best[1]['material'],
            'predicted_comfort': round(best[0], 2),
            'head_height_mm': int(best[1]['head_height']),
            'neck_height_mm': int(best[1]['neck_height']),
            'side_height_mm': int(best[1]['side_height']),
            'hardness': int(best[1]['center_hardness'])
        },
        'all_pillows_ranked': [
            {'id': p['id'], 'brand': p['brand'], 'predicted_comfort': round(score, 2)}
            for score, p in scored
        ],
        'input': {'height': height, 'weight': weight, 'shoulder_width': shoulder_width, 'gender': gender}
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='枕头智能推荐系统 V2')
    parser.add_argument('--height', type=float, help='身高(cm)')
    parser.add_argument('--weight', type=float, help='体重(kg)')
    parser.add_argument('--shoulder', type=float, help='肩宽(cm)')
    parser.add_argument('--neck-curve', type=str, help='颈椎曲度21维数组')
    parser.add_argument('--gender', type=str, default='未知')
    parser.add_argument('--json', action='store_true', help='仅输出JSON')
    args = parser.parse_args()
    
    if not all([args.height, args.weight, args.shoulder, args.neck_curve]):
        print("用法: --height 175 --weight 70 --shoulder 17 --neck-curve '[...]' --gender 男 [--json]")
        exit(1)
    
    neck_curve = json.loads(args.neck_curve)
    result = recommend(args.height, args.weight, args.shoulder, neck_curve, args.gender)
    
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        rp = result['recommended_params']
        p = result['recommended_pillow']
        print(f"推荐参数: 后脑勺高{rp['后脑勺高度_mm']}mm 颈椎高{rp['颈椎高度_mm']}mm 侧睡高{rp['侧睡区高度_mm']}mm 硬度{rp['软硬度']}")
        print(f"推荐枕头: {p['brand']}({p['id']}) 预测舒适度{p['predicted_comfort']} 材质{p['material']}")
        print(f"头高{p['head_height_mm']}mm 颈高{p['neck_height_mm']}mm 侧高{p['side_height_mm']}mm 硬度{p['hardness']}")
        print(f"\n排名Top5:")
        for i, item in enumerate(result['all_pillows_ranked'][:5], 1):
            m = " ★" if i==1 else ""
            print(f"  {i}. {item['brand']} 舒适度{item['predicted_comfort']}{m}")
