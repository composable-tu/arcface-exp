# SPDX-License-Identifier: MulanPSL-2.0

"""
Copyright (c) 2026 composable-tu
This project is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
"""

import time

import cv2
import numpy as np
from openvino import Core


def preprocess_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))
    input_array = np.expand_dims(img, axis=0).astype(np.uint8)
    return input_array


def extract_feature_openvino(image_path, model_xml_path='model/openvino/model.xml'):
    """
    使用OpenVINO模型提取人脸特征向量
    Args:
        image_path: 输入人脸图像路径
        model_xml_path: OpenVINO IR模型XML文件路径

    Returns:
        tuple: (特征向量, 推理时间)
    """
    core = Core()
    model = core.read_model(model=model_xml_path)
    compiled_model = core.compile_model(model=model, device_name="CPU")

    input_data = preprocess_face(image_path)

    start_time = time.time()
    outputs = compiled_model([input_data])
    inference_time = time.time() - start_time

    feature_vector = next(iter(outputs.values()))
    norm = np.linalg.norm(feature_vector)
    return (feature_vector / (norm + 1e-10)).flatten(), inference_time


def main():
    import argparse

    parser = argparse.ArgumentParser(description='提取人脸特征向量 (OpenVINO)')
    parser.add_argument('--image_path', type=str, required=True, help='输入人脸图像路径 (112x112)')
    parser.add_argument('--model_path', type=str, default='model/openvino/model.xml', help='OpenVINO IR模型XML文件路径')

    args = parser.parse_args()

    try:
        # 提取特征向量
        feature_vector, inference_time = extract_feature_openvino(image_path=args.image_path,
                                                                  model_xml_path=args.model_path)

        print(f"特征向量维度: {feature_vector.shape}")
        print(f"特征向量范数: {np.linalg.norm(feature_vector)}")
        print(f"特征向量前10个值: {feature_vector[:10]}")
        print(f"推理时间: {inference_time:.4f} 秒")

        # # 保存特征向量到文件

        # output_path = args.image_path.replace('.jpg', '_feature.npy').replace('.png', '_feature.npy')

        # np.save(output_path, feature_vector)

        # print(f"特征向量已保存到: {output_path}")

    except Exception as e:
        print(f"错误: {str(e)}")


if __name__ == '__main__':
    main()
