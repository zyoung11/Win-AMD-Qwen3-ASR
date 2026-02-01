# 一、创建文件夹

```bat
mkdir Qwen3-ASR
cd Qwen3-ASR
```
<br>

#  二、安装 uv 和 curl（如果没有的话）

```bat
winget install astral-sh.uv
winget install cURL.cURL 
```
<br>

# 三、安装 whl

```bat
curl -L -O https://github.com/zyoung11/Win-AMD-Qwen3-ASR/releases/download/0.1.0/torch-2.8.0a0+gitfc14c65-cp312-cp312-win_amd64.whl 
curl -L -O https://github.com/zyoung11/Win-AMD-Qwen3-ASR/releases/download/0.1.0/torchaudio-2.6.0a0+1a8f621-cp312-cp312-win_amd64.whl 
curl -L -O https://github.com/zyoung11/Win-AMD-Qwen3-ASR/releases/download/0.1.0/torchvision-0.24.0a0+c85f008-cp312-cp312-win_amd64.whl 
```
<br>

# 四、创建虚拟环境并安装

```bat
uv venv --python 3.12.0

uv pip install ./torch-2.8.0a0+gitfc14c65-cp312-cp312-win_amd64.whl ./torchaudio-2.6.0a0+1a8f621-cp312-cp312-win_amd64.whl ./torchvision-0.24.0a0+c85f008-cp312-cp312-win_amd64.whl

uv pip install ./qwen_asr-0.0.6-py3-none-any.whl

uv pip install -U modelscope
```
<br>

# 五、选择要下载的模型

## 1.  Qwen3-ASR-1.7B

```bat
uv run modelscope download --model Qwen/Qwen3-ASR-1.7B  --local_dir ./Qwen3-ASR-1.7B
```

## 2.  Qwen3-ASR-0.6B

```bat
modelscope download --model Qwen/Qwen3-ASR-1.7B  --local_dir ./Qwen3-ASR-0.6B
```

## 3.  Qwen3-ForcedAligner-0.6B

```bat
modelscope download --model Qwen/Qwen3-ASR-1.7B  --local_dir ./Qwen3-ForcedAligner-0.6B
```
<br>

# 六、启动 WebUI

## 1.  Qwen3-ASR-1.7B

```bat
uv run qwen-asr-demo --asr-checkpoint ./Qwen3-ASR-1.7B --ip 0.0.0.0 --port 8000
```

## 2.  Qwen3-ASR-0.6B

```bat
uv run qwen-asr-demo --asr-checkpoint ./Qwen3-ASR-0.6B --ip 0.0.0.0 --port 8000
```

## 3.  Qwen3-ForcedAligner-0.6B

```bat
uv run qwen-asr-demo --asr-checkpoint ./Qwen3-ForcedAligner-0.6B --ip 0.0.0.0 --port 8000
```
<br>

# 七、打开 WebUI [`http://127.0.0.1:8000`](http://127.0.0.1:8000)

