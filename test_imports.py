try:
    import torch
    print(f"PyTorch found: version {torch.__version__}")
except ImportError as e:
    print(f"PyTorch import error: {e}")

try:
    from stable_baselines3 import PPO, A2C, SAC
    print("stable-baselines3 imported successfully")
except ImportError as e:
    print(f"stable-baselines3 import error: {e}")