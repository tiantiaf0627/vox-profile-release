## Vox-Profile: A Speech Foundation Model Benchmark for Characterizing Diverse Speaker and Speech Traits

<p align="center">
  📄 <a href="https://openreview.net/pdf?id=qmGQoZfkEN"><strong>[DMLR Paper]</strong></a> &nbsp;|&nbsp;
  🤗 <a href="https://huggingface.co/collections/tiantiaf/vox-profile-6849966a82005eea95dc6274"><strong>[HuggingFace Models]</strong></a> &nbsp;
  | 🚀 <a href="https://github.com/tiantiaf0627/voxlect/tree/main"><strong>[Voxlect Project]</strong></a> &nbsp;
</p>

#### In this repo, we present Vox-Profile, one of the first benchmarking efforts that systematically evaluate rich multi-dimensional speaker and speech traits from English-speaking voices. Our benchmark is presented below:

<div align="center">
 <img src="img/vox-profile.png" width="800px">
</div>

### Our training data filters output audio shorter than 3 seconds (unreliable predictions) and longer than 15 seconds (computation limitation), so you need to cut your audio to a maximum of 15 seconds, 16kHz and mono channel

### Download Repo
```bash
git clone git@github.com:tiantiaf0627/vox-profile-release.git
```

### Installation
```bash
conda create -n vox_profile python=3.8
cd vox-profile-release
pip install -e .
```


### Quick Example 1 - Whisper Large Narrow Accent
```python
# Load libraries
import torch
import torch.nn.functional as F
from src.model.accent.whisper_accent import WhisperWrapper

# Label List
english_accent_list = [
    'East Asia', 'English', 'Germanic', 'Irish', 
    'North America', 'Northern Irish', 'Oceania', 
    'Other', 'Romance', 'Scottish', 'Semitic', 'Slavic', 
    'South African', 'Southeast Asia', 'South Asia', 'Welsh'
]
    
# Find device
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# Load model from Huggingface
whisper_model = WhisperWrapper.from_pretrained("tiantiaf/whisper-large-v3-narrow-accent").to(device)
whisper_model.eval()

# Load data, here just zeros as the example
# Our training data filters output audio shorter than 3 seconds (unreliable predictions) and longer than 15 seconds (computation limitation)
# So you need to prepare your audio to a maximum of 15 seconds, 16kHz and mono channel
max_audio_length = 15 * 16000
data = torch.zeros([1, 16000]).float().to(device)[:, :max_audio_length]
whisper_logits, whisper_embeddings = whisper_model(data, return_feature=True)
    
# Probability and output
whisper_prob = F.softmax(whisper_logits, dim=1)
print(english_accent_list[torch.argmax(whisper_prob).detach().cpu().item()])
```

|                       | Eastern Asia | English | Germanic | Irish | North America | Northern Irish | Oceania | Other | Romance | Scottish | Semitic | Slavic | South African | South-eastern Asia | Southern Asia | Welsh |
|-----------------------|--------------|---------|----------|-------|----------------|----------------|---------|-------|---------|----------|---------|--------|----------------|---------------------|---------------|-------|
| Eastern Asia          | 89.67        | 1.03    | 0.96     | 0.11  | 2.20           | 0.00           | 0.75    | 0.21  | 2.63    | 0.07     | 0.28    | 0.25   | 0.04           | 1.49                | 0.07          | 0.25  |
| English               | 0.16         | 84.31   | 0.08     | 0.08  | 1.36           | 0.36           | 1.24    | 0.12  | 2.32    | 3.23     | 0.84    | 0.12   | 1.32           | 0.28                | 0.44          | 3.75  |
| Germanic              | 0.11         | 0.00    | 65.76    | 0.00  | 0.11           | 0.00           | 0.00    | 5.99  | 11.75   | 0.00     | 0.23    | 16.05  | 0.00           | 0.00                | 0.00          | 0.00  |
| Irish                 | 0.15         | 5.61    | 1.21     | 47.04 | 8.50           | 0.30           | 2.58    | 0.30  | 8.95    | 23.07    | 0.46    | 0.15   | 1.21           | 0.00                | 0.15          | 0.30  |
| North America         | 2.11         | 1.54    | 1.02     | 0.90  | 67.93          | 0.00           | 1.98    | 0.38  | 9.73    | 0.38     | 5.38    | 0.64   | 2.43           | 1.98                | 3.52          | 0.06  |
| Northern Irish        | 0.00         | 0.00    | 0.00     | 0.00  | 0.00           | 97.50          | 0.00    | 0.00  | 0.00    | 2.50     | 0.00    | 0.00   | 0.00           | 0.00                | 0.00          | 0.00  |
| Oceania               | 0.82         | 4.09    | 4.29     | 0.41  | 7.36           | 0.00           | 64.62   | 0.20  | 10.22   | 0.61     | 0.41    | 2.66   | 3.68           | 0.61                | 0.00          | 0.00  |
| Other                 | 1.40         | 0.59    | 1.40     | 0.16  | 0.70           | 0.00           | 0.05    | 80.99 | 3.54    | 0.00     | 2.42    | 7.04   | 0.48           | 0.27                | 0.54          | 0.43  |
| Romance               | 2.12         | 0.00    | 1.73     | 0.20  | 0.43           | 0.00           | 0.00    | 0.70  | 89.08   | 0.00     | 1.29    | 4.15   | 0.00           | 0.23                | 0.07          | 0.00  |
| Scottish              | 0.00         | 5.18    | 0.00     | 10.36 | 5.18           | 0.36           | 1.61    | 0.00  | 0.18    | 75.54    | 0.18    | 0.00   | 0.89           | 0.00                | 0.18          | 0.36  |
| Semitic               | 31.54        | 0.00    | 0.05     | 0.00  | 0.00           | 0.00           | 0.00    | 0.11  | 10.43   | 0.00     | 57.35   | 0.37   | 0.00           | 0.11                | 0.05          | 0.00  |
| Slavic                | 0.83         | 0.00    | 5.82     | 0.00  | 0.00           | 0.00           | 0.00    | 5.54  | 16.07   | 0.00     | 1.66    | 70.08  | 0.00           | 0.00                | 0.00          | 0.00  |
| South African         | 0.00         | 4.00    | 2.00     | 0.00  | 4.50           | 0.00           | 8.50    | 2.50  | 2.00    | 1.50     | 0.00    | 0.00   | 68.00          | 0.00                | 4.00          | 3.00  |
| South-eastern Asia    | 41.10        | 0.57    | 0.08     | 0.12  | 2.53           | 0.00           | 0.37    | 0.00  | 3.54    | 0.00     | 0.12    | 0.04   | 0.73           | 49.12               | 1.67          | 0.00  |
| Southern Asia         | 0.62         | 0.22    | 1.02     | 0.04  | 0.66           | 0.00           | 0.09    | 0.31  | 6.16    | 0.00     | 1.02    | 0.18   | 0.62           | 0.09                | 88.88         | 0.09  |
| Welsh                 | 0.00         | 26.89   | 0.00     | 0.20  | 0.20           | 0.00           | 0.20    | 0.20  | 0.10    | 3.58     | 0.10    | 0.00   | 0.82           | 0.00                | 0.00          | 67.69 |

### Quick Example 2 - WavLM Large Narrow Accent
```python
# Load libraries
import torch
import torch.nn.functional as F
from src.model.accent.wavlm_accent import WavLMWrapper

# Label List
english_accent_list = [
    'East Asia', 'English', 'Germanic', 'Irish', 
    'North America', 'Northern Irish', 'Oceania', 
    'Other', 'Romance', 'Scottish', 'Semitic', 'Slavic', 
    'South African', 'Southeast Asia', 'South Asia', 'Welsh'
]
    
# Find device
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# Load model from Huggingface
wavlm_model = WavLMWrapper.from_pretrained("tiantiaf/wavlm-large-narrow-accent").to(device)
wavlm_model.eval()

# Load data, here just zeros as the example
# Our training data filters output audio shorter than 3 seconds (unreliable predictions) and longer than 15 seconds (computation limitation)
# So you need to prepare your audio to a maximum of 15 seconds, 16kHz and mono channel
max_audio_length = 15 * 16000
data = torch.zeros([1, 16000]).float().to(device)[:, :max_audio_length]
wavlm_logits, wavlm_embeddings = wavlm_model(data, return_feature=True)
    
# Probability and output
wavlm_prob = F.softmax(wavlm_logits, dim=1)
print(english_accent_list[torch.argmax(wavlm_prob).detach().cpu().item()])
```

#### Given that the Vox-Profile Benchmark paper is still under peer-review, we provide limited set of models and model weights before the review is concluded. But below are the models we currently put out.

### WavLM-Large Models

 Model Name  | Data  | Pre-trained Model | Use LoRa |  LoRa Rank Size  | Output | Example Code |
|--------------------------------------------------------|-------|-----------------|-----------------|-------------|------------------------|------------------------|
| [wavlm-large-sex-age](https://huggingface.co/tiantiaf/wavlm-large-age-sex)   | CommonVoice+Timit+Voxceleb (age enriched) | wavlm-large              | Yes              | 16              | Sex (2-class) / Age (0-1)*100 Years  | [Example](https://github.com/tiantiaf0627/vox-profile-release/blob/main/src/example/age_sex_wavlm.py) |
| [wavlm-large-broader-accent](https://huggingface.co/tiantiaf/wavlm-large-broader-accent)   | See Paper (11 Datasets) | wavlm-large              | Yes              | 16              | North American / British / Other (3-class)  | [Example](https://github.com/tiantiaf0627/vox-profile-release/blob/main/src/example/broad_accent_wavlm.py) |
| [wavlm-large-narrow-accent](https://huggingface.co/tiantiaf/wavlm-large-narrow-accent)   | See Paper (11 Datasets) | wavlm-large              | Yes              | 16              | See example  | [Example](https://github.com/tiantiaf0627/vox-profile-release/blob/main/src/example/narrow_accent_wavlm.py) |
| [wavlm-large-voice-quality](https://huggingface.co/tiantiaf/wavlm-large-voice-quality)   | ParaSpeechCaps | wavlm-large              | Yes              | 16              | See example  | [Example](https://github.com/tiantiaf0627/vox-profile-release/blob/main/src/example/voice_quality_wavlm.py) |
| [wavlm-large-influency](https://huggingface.co/tiantiaf/wavlm-large-speech-flow)   | SEP28K+FluencyBank | wavlm-large              | Yes              | 16              | Fluent/Disfluent (Specified Disfluency Types)  | [Example](https://github.com/tiantiaf0627/vox-profile-release/blob/main/src/example/fluency_wavlm.py) |
| [wavlm-large-categorical-emotion](https://huggingface.co/tiantiaf/wavlm-large-categorical-emotion)   | MSP-Podcast | wavlm-large              | No              | NA              | 8 Emotions + Other  | [Example](https://github.com/tiantiaf0627/vox-profile-release/blob/main/src/example/categorized_emotion_wavlm.py) |
| [wavlm-large-dim-emotion](https://huggingface.co/tiantiaf/tiantiaf/wavlm-large-msp-podcast-emotion-dim)   | MSP-Podcast | wavlm-large              | No              | NA              | Arousal/Valence/Dominance  | [Example](https://github.com/tiantiaf0627/vox-profile-release/blob/main/src/example/dim_emotion_wavlm.py) |


### Whisper-Large V3 Models
Model Name  | Data  | Pre-trained Model | Use LoRa |  LoRa Rank Size  | Output | Example Code |
|--------------------------------------------------------|-------|-----------------|-----------------|-------------|------------------------|------------------------|
| [whisper-large-v3-broader-accent](https://huggingface.co/tiantiaf/whisper-large-v3-broad-accent)   | See Paper (11 Datasets) | whisper-large v3              | Yes              | 16              | North American / British / Other (3-class)  | [Example](https://github.com/tiantiaf0627/vox-profile-release/blob/main/src/example/broad_accent_whisper.py) |
| [whisper-large-v3-narrow-accent](https://huggingface.co/tiantiaf/whisper-large-v3-narrow-accent)   | See Paper (11 Datasets) | whisper-large v3             | Yes              | 16              | See example  | [Example](https://github.com/tiantiaf0627/vox-profile-release/blob/main/src/example/narrow_accent_whisper.py) |
| [whisper-large-v3-voice-quality](https://huggingface.co/tiantiaf/whisper-large-v3-voice-quality)   | ParaSpeechCaps | whisper-large v3              | Yes              | 16              | See example  | [Example](https://github.com/tiantiaf0627/vox-profile-release/blob/main/src/example/voice_quality_whisper.py) |
| [whisper-large-v3-influency](https://huggingface.co/tiantiaf/whisper-large-v3-speech-flow)   | SEP28K+FluencyBank | whisper-large v3              | Yes              | 16              | Fluent/Disfluent (Specified Disfluency Types)  | [Example](https://github.com/tiantiaf0627/vox-profile-release/blob/main/src/example/fluency_whisper.py) |
| [whisper-large-v3-categorical-emotion](https://huggingface.co/tiantiaf/whisper-large-v3-msp-podcast-emotion)   | MSP-Podcast | whisper-large v3             | Yes              | 16              | 8 Emotions + Other  | [Example](https://github.com/tiantiaf0627/vox-profile-release/blob/main/src/example/categorized_emotion_whisper.py) |
| [whisper-large-v3-dim-emotion](https://huggingface.co/tiantiaf/whisper-large-v3-msp-podcast-emotion-dim)   | MSP-Podcast | whisper-large v3             | No              | NA              | Arousal/Valence/Dominance  | [Example](https://github.com/tiantiaf0627/vox-profile-release/blob/main/src/example/dim_emotion_whisper.py) |


### Labeling Scheme
In Vox-Profile, we experiments with over 15 publicly available datasets to predict static traits (speaker age, speaker sex, speaker accent, and voice quality) and dynamic traits (speech emotion, speech flow, and speech expressiveness) in different recording conditions and elicitation settings (e.g., read, spontaneous, and conversational speech). Our labeling taxonomy is described below:

<div align="center">
 <img src="img/label_mapping.png" width="400px">
</div>

### Enabling Versatile Speech Applications with Vox-Profile

Our Vox-Profile can be used as a fundamental component to support a versatile speech applications.

#### 1. Speech Model Performance Analysis

We generate speaker and speech traits for existing datasets and investigate whether these generated labels can lead to the same insights as using the ground truth trait information in analyzing the speech model performances. For example, in the results below, we identify that regardless of grouping by groud truth traits or traits inferred by Vox-Profile, the ASR performance trends remain the same across different traits.

<div align="center">
 <img src="img/wer.png" width="800px">
</div>

#### 2. Automated Evaluation Tool for Speech Generation Tasks

We demonstrate the utility of Vox-Profile as an evaluation tool for speech generation tasks by comparing two representative models: FreeVC and VALLE-X. As shown in Table below, the accent prediction scores and the cosine similarity for the synthesized samples from FreeVC suggest greater similarity to the source speaker’s accent than to the reference speaker. In contrast, the scores for VALLE-X indicate closer alignment with the reference speaker’s accent in most conditions. These findings are consistent with previous studies, which report that FreeVC has limited capability in replicating the accentual features of the reference speaker compare to VALLE-X.

<div align="center">
 <img src="img/vc_evaluation.png" width="800px">
</div>


#### 3. Generating Synthetic Speaking Style Prompt
Vox-Profile provides a more extensive and varied set of traits, including speech flow, arousal, valence, and speaker age. Moreover, computational models of Vox-Profile output probabilistic predictions for each trait, enabling more nuanced and confidence sensitive descriptions. For example, a Scottish accent prediction with a probability of 0.9 can be described as having a distinct Scottish accent.

<div align="center">
 <img src="img/speaking_prompt.png" width="800px">
</div>

Human-evaluation results comparing synthetic speaking style prompts from Vox-Profile and human-annotated speaking style prompts from ParaSpeechCaps suggest that this group of human raters shows similar preference levels for both synthetic and human-annotated speaking style prompts. Specifically, they favor the emotion, age, and speech flow descriptions generated by Vox-Profile over those from ParaSpeechCaps.

#### Responsible Use: Users should respect the privacy and consent of the data subjects, and adhere to the relevant laws and regulations in their jurisdictions when using Vox-Profile. We also advise the use of Vox-Profile under the RAIL license.

❌ **Exemplar Out-of-Scope Use**
- Clinical or diagnostic applications
- Surveillance
- Privacy-invasive applications
- No commercial use

#### If you like our work or use the models in your work, kindly cite the following. We appreciate your recognition!
```
@article{feng2026vox,
  title={Vox-Profile: A Speech Foundation Model Benchmark for Characterizing Diverse Speaker Traits and Speech Properties},
  author={Feng, Tiantian and Lee, Jihwan and Xu, Anfeng and Lee, Yoonjeong and Lertpetchpun, Thanathai and Shi, Xuan and Wang, Helin and Thebaud, Thomas and Moro-Velazquez, Laureano and Byrd, Dani and others},
  journal={Journal of Data-centric Machine Learning Research},
  year={2026}
}
```
