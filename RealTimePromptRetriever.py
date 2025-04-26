import torch
import numpy as np
import clip
import cv2
import os   
import json
from PIL import Image
from decord import VideoReader, cpu
from typing import List, Tuple


class PromptRetriever:
    def __init__(self, 
                 demo_database_path,
                 text_weight=0.6, 
                 top_k=5, 
                 video_frame_rate=2, 
                 aggregation_method='mean'):
        """
        提示检索器
        
        参数:
        - demo_database_path: demo数据库路径，包含<text,video>格式的示例
        - text_weight: 文本相似度的权重
        - top_k: 返回的最相关示例数量
        - video_frame_rate: 视频采样帧率
        - aggregation_method: 视频特征聚合方法
        """
        self.text_weight = text_weight
        self.top_k = top_k
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # 加载CLIP模型
        print("Loading CLIP model...")
        self.model, self.preprocess = clip.load('ViT-B/32', self.device)
        
        # 视频处理参数
        self.video_frame_rate = video_frame_rate
        self.aggregation_method = aggregation_method
        
        # 加载demo数据库
        self.demo_examples = self._load_demo_database(demo_database_path)
        print(f"Loaded {len(self.demo_examples)} examples from demo database")
        
        # 预计算demo示例的特征
        self._precompute_demo_features()
    
    def _load_demo_database(self, database_path):
        """加载demo数据库中的示例"""
        examples = []
        
        # 假设数据库是一个包含JSON文件的目录，每个JSON文件描述一个示例
        for filename in os.listdir(database_path):
            if filename.endswith('.json'):
                with open(os.path.join(database_path, filename), 'r') as f:
                    example_data = json.load(f)
                    
                    # 假设每个示例包含text, video_path和response字段
                    text = example_data['text']
                    video_path = os.path.join(database_path, example_data['video_path'])
                    response = example_data['response']
                    
                    examples.append((text, video_path, response))
        
        return examples
    
    def _precompute_demo_features(self):
        """预计算demo示例的特征以提高性能"""
        print("Precomputing demo example features...")
        self.demo_text_features = []
        self.demo_video_features = []
        
        for text, video_path, _ in self.demo_examples:
            # 编码文本
            text_feature = self.encode_text(text)
            self.demo_text_features.append(text_feature)
            
            # 编码视频
            video_feature = self.encode_video(video_path)
            self.demo_video_features.append(video_feature)
        
        # 将特征转换为张量以便于批处理计算
        self.demo_text_features = torch.cat(self.demo_text_features, dim=0)
        self.demo_video_features = torch.cat(self.demo_video_features, dim=0)
        print("Precomputation completed")
    
    def encode_text(self, text):
        """编码文本"""
        text_tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
        return text_features / text_features.norm(dim=-1, keepdim=True)
    
    def encode_image(self, image_path_or_array):
        """编码单个图像"""
        if isinstance(image_path_or_array, str):
            # 从文件加载图像
            image = Image.open(image_path_or_array).convert('RGB')
        elif isinstance(image_path_or_array, np.ndarray):
            # 转换OpenCV格式(BGR)到PIL格式(RGB)
            image = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        else:
            # 假设已经是PIL图像
            image = image_path_or_array
        
        processed_image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(processed_image)
        return image_features / image_features.norm(dim=-1, keepdim=True)
    
    def encode_video(self, video_path):
        """编码视频"""
        # 使用decord读取视频
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        
        # 计算采样间隔
        sample_interval = max(1, int(fps / self.video_frame_rate))
        frame_indices = list(range(0, len(vr), sample_interval))
        
        if not frame_indices:
            return torch.zeros((1, 512), device=self.device)
            
        # 获取采样帧
        frames = vr.get_batch(frame_indices).asnumpy()
        
        # 编码每一帧
        frame_features = []
        for frame in frames:
            # 转换为PIL图像
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame)
            frame_feature = self.encode_image(pil_frame)
            frame_features.append(frame_feature)
        
        frame_features = torch.cat(frame_features, dim=0)
        
        # 根据选择的方法聚合特征
        if self.aggregation_method == 'mean':
            video_features = torch.mean(frame_features, dim=0, keepdim=True)
        elif self.aggregation_method == 'max':
            video_features, _ = torch.max(frame_features, dim=0, keepdim=True)
        elif self.aggregation_method == 'attention':
            # 简单的自注意力机制
            attn_weights = torch.matmul(frame_features, frame_features.T)
            attn_weights = torch.softmax(attn_weights, dim=1)
            video_features = torch.matmul(attn_weights, frame_features)
            video_features = torch.mean(video_features, dim=0, keepdim=True)
        
        return video_features / video_features.norm(dim=-1, keepdim=True)
    
    def retrieve(self, text_instruction, image_path_or_array):
        """
        检索与当前图像和文本指令最相关的示例
        
        参数:
        - text_instruction: 当前任务的文本指令
        - image_path_or_array: 图像路径、numpy数组或PIL图像
        
        返回:
        - 最相关的top_k个示例
        """
        # 编码当前文本指令
        query_text_features = self.encode_text(text_instruction)
        
        # 编码当前图像
        query_image_features = self.encode_image(image_path_or_array)
        
        # 计算文本相似度 (1 x N)
        text_similarities = torch.matmul(
            query_text_features, self.demo_text_features.T).squeeze(0)
        
        # 计算图像-视频相似度 (1 x N)
        visual_similarities = torch.matmul(
            query_image_features, self.demo_video_features.T).squeeze(0)
        
        # 计算总分数
        scores = visual_similarities + self.text_weight * text_similarities
        
        # 选择top-k
        top_indices = torch.topk(scores, min(self.top_k, len(scores)), dim=0).indices.cpu().numpy()
        
        # 返回最相关的示例
        return [self.demo_examples[i] for i in top_indices]
    
    def batch_retrieve(self, text_instructions, image_paths_or_arrays):
        """
        批量检索与多个图像和文本指令最相关的示例
        
        参数:
        - text_instructions: 文本指令列表
        - image_paths_or_arrays: 图像路径或数组列表
        
        返回:
        - 每个查询对应的最相关示例列表
        """
        results = []
        for text, image in zip(text_instructions, image_paths_or_arrays):
            results.append(self.retrieve(text, image))
        return results
