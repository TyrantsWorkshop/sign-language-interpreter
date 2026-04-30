# Installation & Troubleshooting

## Common Issues

### 1. CUDA Out of Memory

**Error:** `CUDA out of memory`

**Solution:**
```bash
# Reduce batch size
export BATCH_SIZE=16

# Or use CPU
export DEVICE=cpu
```

### 2. WebSocket Connection Failed

**Error:** `WebSocket is closed before the connection is established`

**Solution:**
- Ensure backend is running on correct port
- Check CORS configuration
- Verify WebSocket URL in frontend .env

### 3. Camera Permission Denied

**Error:** `NotAllowedError: Permission denied`

**Solution:**
- Grant camera permission in browser settings
- Use HTTPS (required for some browsers)
- Check browser camera settings

### 4. Model Loading Failed

**Error:** `FileNotFoundError: saved_models/sign_language_model.pt`

**Solution:**
```bash
mkdir -p backend/saved_models
# Download or train models
python training/train_sign_model.py --dataset ./data
```

### 5. MediaPipe Detection Issues

**Problem:** Poor hand detection

**Solution:**
- Improve lighting
- Move closer to camera (1-2 meters)
- Reduce background clutter
- Check MediaPipe complexity setting

## GPU Setup

### NVIDIA
```bash
# Check CUDA installation
char --version

# Install cuDNN
wget https://developer.nvidia.com/cudnn
```

### AMD
```bash
# Install ROCm
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
sudo apt update
sudo apt install rocm-dkms
```

## Performance Optimization

### Mixed Precision Training
```python
from torch.cuda.amp import autocast

with autocast():
    loss = model(x)
    loss.backward()
```

### Model Quantization
```python
import torch.quantization as quant

quantized_model = quant.quantize_dynamic(
    model, 
    {nn.Linear, nn.Conv2d},
    dtype=torch.qint8
)
```

## Deployment Checklist

- [ ] Update .env with production keys
- [ ] Set DEBUG=False
- [ ] Configure CORS properly
- [ ] Set up SSL/HTTPS
- [ ] Enable logging
- [ ] Test all endpoints
- [ ] Load test the application
- [ ] Set up monitoring
- [ ] Configure backup/recovery
- [ ] Document API changes

## Memory Requirements

| Component | Size |
|-----------|------|
| Sign Model | 34MB |
| Emotion Model | 8.5MB |
| Gesture Model | 0.33MB |
| Runtime (PyTorch) | ~2GB |
| **Total** | **~2.1GB** |

## Network Requirements

- Minimum: 1 Mbps (for video streaming)
- Recommended: 10 Mbps
- For cloud: Use h.264 compression
