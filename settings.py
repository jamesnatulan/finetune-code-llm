DATASET = "smangrul/hf-stack-v1"
DATA_COLUMN = "content"
MAX_SEQ_LENGTH = 2048
DTYPE = None  # Set to None for autodetection. Float16 for Tesla T4, v100, Bfloat16 for Ampere+
PARAM_SIZE = "1.5"  # Model parameter size, can be 0.5, 1.5, 3, 7, 14, 32
BASE_OUTPUT_DIR = "runs"
RESUME_FROM_CHECKPOINT = None # Set this to a checkpoint path to resume training from that checkpoint e.g. runs/run2
SEED = 42 # Seed for reproducibility

# Training parameters
MAX_STEPS = 2000  # max_steps
BATCH_SIZE = 16  # batch_size
GR_ACC_STEPS = 1  # gradient_accumulation_steps
LR = 5e-4  # learning_rate
LR_SCHEDULER_TYPE = "cosine"  # lr_scheduler_type
WEIGHT_DECAY = 0.01  # weight_decay
NUM_WARMUP_STEPS = 30  # num_warmup_steps
EVAL_FREQ = 100  # eval_freq
SAVE_FREQ = 100  # save_freq
LOG_FREQ = 25 # logging frequence, uses tensorboard logging

# PEFT parameters
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0

# FIM trasformations arguments
FIM_RATE = 0.5  # fim_rate
FIM_SPM_RATE = 0.5  # fim_spm_rate

# GGUF Options
# Quantization method for GGUF
# Supported quantizations:
# q4_0, q4_1, q5_0, q5_1, q8_0
# q3_k_s, q3_k_m, q3_k_l, q4_k_s, q4_k_m, q5_k_s, q5_k_m, q6_k
GGUF_QUANT_METHOD = "q4_k_m" 
