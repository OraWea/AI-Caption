import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'course-project-secret-key'
    UPLOAD_FOLDER = 'static/uploads'
    OUTPUT_FOLDER = 'output'
    
    # =====================================================
    # ML Model Configuration (Workload Highlight)
    # =====================================================
    
    # 1. Whisper ASR Model
    WHISPER_MODEL = 'medium' 
    WHISPER_DEVICE = 'cuda' # 若有N卡改为 'cuda'
    
    # 2. Neural Machine Translation (NMT) Model
    # 使用 Meta 的 NLLB-200-Distilled (600M参数，兼顾性能和效果)
    # HuggingFace ID: facebook/nllb-200-distilled-600M
    #NMT_MODEL_ID = "facebook/nllb-200-distilled-600M"
    NMT_MODEL_ID = "facebook/nllb-200-1.3B"
    # 3. Reflection LLM Model (用于反思改进)
    # 使用 Qwen2.5-0.5B-Instruct (超轻量级指令模型)
    # HuggingFace ID: Qwen/Qwen2.5-0.5B-Instruct
    #REFLECTION_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
    REFLECTION_MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
    # 4. Transfer Learning Config (占位符配置)
    FINETUNE_EPOCHS = 3
    FINETUNE_LEARNING_RATE = 2e-5
    FINETUNE_BATCH_SIZE = 16

    # 5. Quality Estimation (QE) Model
    # 使用 Bi-Encoder (Sentence-BERT) 进行翻译质量评估
    # 模型: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (更稳定，下载量大)
    QE_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    QE_THRESHOLD = 0.7
    ENABLE_QE = True
    
    # =====================================================
    
    # 功能开关
    ENABLE_REFLECTION = True  # 开启反思模式
    
    # 其他配置保持不变
    SUPPORTED_FORMATS = ['.mp3', '.wav', '.m4a', '.flac', '.mp4', '.mkv']
    SUBTITLE_FORMATS = ['srt', 'vtt']
    TEMP_FOLDER = 'temp'
    
    @staticmethod
    def init_app(app):
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(Config.TEMP_FOLDER, exist_ok=True)