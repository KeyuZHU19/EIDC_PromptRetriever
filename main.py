from RealTimePromptRetriever import PromptRetriever
import os


# 初始化检索器
retriever = PromptRetriever(
    demo_database_path="/path/to/demo_database",
    text_weight=0.6,
    top_k=1,
    video_frame_rate=2,
    aggregation_method='attention'
)

# 单个查询示例
image_path = "input_image.jpg"
with open('instruction.txt', 'r') as f:
    text_instruction = f.read().strip()

# 检索相关示例
matched_examples = retriever.retrieve(text_instruction, image_path)

# 处理结果
print(f"Found {len(matched_examples)} matching examples:")
for i, (text, video_path, response) in enumerate(matched_examples):
    print(f"\nExample {i+1}:")
    print(f"Text: {text}")
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Response: {response[:100]}..." if len(response) > 100 else f"Response: {response}")

# 批量查询示例
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
text_instructions = ["指令1", "指令2", "指令3"]

batch_results = retriever.batch_retrieve(text_instructions, image_paths)
