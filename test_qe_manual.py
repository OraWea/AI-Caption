import logging
from models.quality_estimator import QualityEstimator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_qe():
    print("Testing Quality Estimator...")
    
    try:
        # 初始化模型
        qe = QualityEstimator(device='cpu') # 测试时使用 CPU
        
        # 测试用例
        test_cases = [
            ("Hello world", "Bonjour le monde", "High Quality (EN->FR)"),
            ("Hello world", "Hallo Welt", "High Quality (EN->DE)"),
            ("Hello world", "你好世界", "High Quality (EN->ZH)"),
            ("Hello world", "Goodbye everyone", "Low Quality (Wrong Meaning)"),
            ("Hello world", "今天天气不错", "Low Quality (Unrelated)"),
        ]
        
        print(f"\n{'='*50}")
        print(f"{'Source':<20} | {'Translation':<20} | {'Score':<10} | {'Expected'}")
        print(f"{'-'*50}")
        
        for src, trans, desc in test_cases:
            score = qe.estimate(src, trans)
            print(f"{src:<20} | {trans:<20} | {score:.4f}     | {desc}")
            
        print(f"{'='*50}\n")
        
        # 批量测试
        print("Testing Batch Estimation...")
        pairs = [[src, trans] for src, trans, _ in test_cases]
        scores = qe.estimate_batch(pairs)
        print(f"Batch Scores: {scores}")
        
        print("\n✅ Test Passed!")
        
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")

if __name__ == "__main__":
    test_qe()
