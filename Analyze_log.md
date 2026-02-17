[Step 1] Analyzing music...
[MusicAnalyzer] Analyzing: outputs/mv_8c487a68/music/들꽃의_서
약_업.mp3
  Duration: 323.24s
  Segments: 24
[MusicAnalyzer] Analysis complete

[Step 1.1] Separating vocals with Demucs...
  [Demucs] Separating vocals from: 들꽃의_서약_업.mp3 (device=cuda)
  [Demucs] Vocals saved: outputs/mv_8c487a68\music\vocals.wav

[Step 1.5] User lyrics received (961 chars)

[Step 1.6] WhisperX forced alignment...
  [WhisperX] Device: cuda
  [WhisperX] Step 1: Whisper STT...
C:\Project\.venv\lib\site-packages\pyannote\audio\core\io.py:47: UserWarning:
torchcodec is not installed correctly so built-in audio decoding will fail. Solutions are:
* use audio preloaded in-memory as a {'waveform': (channel, time) torch.Tensor, 'sample_rate': int} dictionary;
* fix torchcodec installation. Error message was:

Could not load libtorchcodec. Likely causes:
          1. FFmpeg is not properly installed in your environment. We support
             versions 4, 5, 6, 7, and 8, and we attempt to load libtorchcodec
             for each of those versions. Errors for versions not installed on
             your system are expected; only the error for your installed FFmpeg
             version is relevant. On Windows, ensure you've installed the
             "full-shared" version which ships DLLs.
          2. The PyTorch version (2.7.1+cu118) is not compatible with
             this version of TorchCodec. Refer to the version 
compatibility
             table:
             https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec.
          3. Another runtime dependency; see exceptions below.
        The following exceptions were raised as we tried to load libtorchcodec:

[start of libtorchcodec loading traceback]
FFmpeg version 8:
Traceback (most recent call last):
  File "C:\Project\.venv\lib\site-packages\torchcodec\_core\ops.py", line 57, in load_torchcodec_shared_libraries
    torch.ops.load_library(core_library_path)
  File "C:\Project\.venv\lib\site-packages\torch\_ops.py", line 1392, in load_library
    ctypes.CDLL(path)
  File "C:\Users\twins\AppData\Local\Programs\Python\Python310\lib\ctypes\__init__.py", line 374, in __init__
    self._handle = _dlopen(self._name, mode)
FileNotFoundError: Could not find module 'C:\Project\.venv\Lib\site-packages\torchcodec\libtorchcodec_core8.dll' (or one of 
its dependencies). Try using the full path with constructor syntax.

FFmpeg version 7:
Traceback (most recent call last):
  File "C:\Project\.venv\lib\site-packages\torchcodec\_core\ops.py", line 57, in load_torchcodec_shared_libraries
    torch.ops.load_library(core_library_path)
  File "C:\Project\.venv\lib\site-packages\torch\_ops.py", line 1392, in load_library
    ctypes.CDLL(path)
  File "C:\Users\twins\AppData\Local\Programs\Python\Python310\lib\ctypes\__init__.py", line 374, in __init__
    self._handle = _dlopen(self._name, mode)
FileNotFoundError: Could not find module 'C:\Project\.venv\Lib\site-packages\torchcodec\libtorchcodec_core7.dll' (or one of 
its dependencies). Try using the full path with constructor syntax.

FFmpeg version 6:
Traceback (most recent call last):
  File "C:\Project\.venv\lib\site-packages\torchcodec\_core\ops.py", line 57, in load_torchcodec_shared_libraries
    torch.ops.load_library(core_library_path)
  File "C:\Project\.venv\lib\site-packages\torch\_ops.py", line 1392, in load_library
    ctypes.CDLL(path)
  File "C:\Users\twins\AppData\Local\Programs\Python\Python310\lib\ctypes\__init__.py", line 374, in __init__
    self._handle = _dlopen(self._name, mode)
FileNotFoundError: Could not find module 'C:\Project\.venv\Lib\site-packages\torchcodec\libtorchcodec_core6.dll' (or one of 
its dependencies). Try using the full path with constructor syntax.

FFmpeg version 5:
Traceback (most recent call last):
  File "C:\Project\.venv\lib\site-packages\torchcodec\_core\ops.py", line 57, in load_torchcodec_shared_libraries
    torch.ops.load_library(core_library_path)
  File "C:\Project\.venv\lib\site-packages\torch\_ops.py", line 1392, in load_library
    ctypes.CDLL(path)
  File "C:\Users\twins\AppData\Local\Programs\Python\Python310\lib\ctypes\__init__.py", line 374, in __init__
    self._handle = _dlopen(self._name, mode)
FileNotFoundError: Could not find module 'C:\Project\.venv\Lib\site-packages\torchcodec\libtorchcodec_core5.dll' (or one of 
its dependencies). Try using the full path with constructor syntax.

FFmpeg version 4:
Traceback (most recent call last):
  File "C:\Project\.venv\lib\site-packages\torchcodec\_core\ops.py", line 57, in load_torchcodec_shared_libraries
    torch.ops.load_library(core_library_path)
  File "C:\Project\.venv\lib\site-packages\torch\_ops.py", line 1392, in load_library
    ctypes.CDLL(path)
  File "C:\Users\twins\AppData\Local\Programs\Python\Python310\lib\ctypes\__init__.py", line 374, in __init__
    self._handle = _dlopen(self._name, mode)
FileNotFoundError: Could not find module 'C:\Project\.venv\Lib\site-packages\torchcodec\libtorchcodec_core4.dll' (or one of 
its dependencies). Try using the full path with constructor syntax.
[end of libtorchcodec loading traceback].
  warnings.warn(
2026-02-14 20:44:50 - whisperx.vads.pyannote - INFO - Performing voice activity detection using Pyannote...
Lightning automatically upgraded your loaded checkpoint from v1.5.4 to v2.6.1. To apply the upgrade to your files permanently, run `python -m lightning.pytorch.utilities.upgrade_checkpoint C:\Project\.venv\lib\site-packages\whisperx\assets\pytorch_model.bin`
C:\Project\.venv\lib\site-packages\pyannote\audio\utils\reproducibility.py:74: ReproducibilityWarning: TensorFloat-32 (TF32) has been disabled as it might lead to reproducibility issues 
and lower accuracy.
It can be re-enabled by calling
   >>> import torch
   >>> torch.backends.cuda.matmul.allow_tf32 = True
   >>> torch.backends.cudnn.allow_tf32 = True
See https://github.com/pyannote/pyannote-audio/issues/1370 for more details.

  warnings.warn(
  [WhisperX] Forced alignment failed: Library cublas64_12.dll 
is not found or cannot be loaded
Traceback (most recent call last):
  File "C:\Project\storycut\agents\mv_pipeline.py", line 191, 
in _forced_align_whisperx
    stt_result = stt_model.transcribe(audio, language=language, batch_size=16)
  File "C:\Project\.venv\lib\site-packages\whisperx\asr.py", line 259, in transcribe
    for idx, out in enumerate(self.__call__(data(audio, vad_segments), batch_size=batch_size, num_workers=num_workers)):    
  File "C:\Project\.venv\lib\site-packages\transformers\pipelines\pt_utils.py", line 126, in __next__
    item = next(self.iterator)
  File "C:\Project\.venv\lib\site-packages\transformers\pipelines\pt_utils.py", line 127, in __next__
    processed = self.infer(item, **self.params)
  File "C:\Project\.venv\lib\site-packages\transformers\pipelines\base.py", line 1374, in forward
    model_outputs = self._forward(model_inputs, **forward_params)
  File "C:\Project\.venv\lib\site-packages\whisperx\asr.py", line 163, in _forward
    outputs = self.model.generate_segment_batched(model_inputs['inputs'], self.tokenizer, self.options)
  File "C:\Project\.venv\lib\site-packages\whisperx\asr.py", line 60, in generate_segment_batched
    encoder_output = self.encode(features)
  File "C:\Project\.venv\lib\site-packages\whisperx\asr.py", line 97, in encode
    return self.model.encode(features, to_cpu=to_cpu)
RuntimeError: Library cublas64_12.dll is not found or cannot be loaded
  [WhisperX] Alignment failed - subtitles will use even distribution
  Duration: 323.2s
  BPM: N/A
  Segments: 24
  Lyrics: YES (961 chars)

[BG-STT] Pre-caching Gemini Audio STT...
INFO:     127.0.0.1:64789 - "POST /api/mv/upload HTTP/1.1" 200 OK
[MV Pipeline] Loading project: mv_8c487a68
[MV Pipeline] manifest exists: True
[Subtitle Test Thread] Starting for mv_8c487a68
INFO:     127.0.0.1:64810 - "POST /api/mv/subtitle-test/mv_8c487a68 HTTP/1.1" 200 OK
  [SubTest] Raw lyrics length: 961 chars
  [SubTest] Raw lyrics newlines: 96 LF, 96 CR
  [SubTest] Raw lyrics preview (repr): '[Intro]\r\n안개가 들판
을 덮고\r\n종소리 멀어져 가면\r\n나는 들꽃 한 송이를 쥐고\r\n 
너의 이름을 삼킨다\r\n\r\n[Verse 1]\r\n성벽의 그림자 아래\r\n 
그는 전쟁에서 돌아와\r\n갑옷 틈새로 새는 겨울\r\n눈빛에 박힌  
먼지\r\n\r\n“괜찮아” 말했지만\r\n숨은 떨리고\r\n나는 그의 손을
 잡아\r\n따뜻함을 나눠 줘\r\n\r\n[Pre-Chorus]\r\n마을의 좁은  
길 끝\r\n그가 멈춰 서서\r\n내 이마 위에 조용히\r\n반지 대신 서
약을 얹어\r\n\r\n[Chorus]\r\n그대여, 돌아와 줘\r\n돌아와 줘\r\n들꽃이 지지 않게\r\n내 이름을 불러 줘\r\n불러 줘\r\n세상이 그
'
  [SubTest] Lyrics: 74 lines (after filtering)
  [WhisperX] Device: cuda
  [WhisperX] Step 1: Whisper STT...
2026-02-14 20:44:53 - whisperx.vads.pyannote - INFO - Performing voice activity detection using Pyannote...
Lightning automatically upgraded your loaded checkpoint from v1.5.4 to v2.6.1. To apply the upgrade to your files permanently, run `python -m lightning.pytorch.utilities.upgrade_checkpoint C:\Project\.venv\lib\site-packages\whisperx\assets\pytorch_model.bin`
  [Gemini-STT] File uploaded: files/j4osflbqpncd
  [Gemini-STT] Extracting vocal segments...
[MV Pipeline] Loading project: mv_8c487a68
[MV Pipeline] manifest exists: True