## Figuring out how to use data
###  1. Dynamic annotations v/s static annotations:
Dreferring dynamic so we have more training data, starting from 15s, with 5second clip size.

Initital df structure.
<img width="767" height="233" alt="Screenshot from 2025-12-12 02-54-47" src="https://github.com/user-attachments/assets/6c3c807b-0065-42ba-98f3-979a1ccee0ed" />

Final df structure:

<img width="426" height="210" alt="Screenshot from 2025-12-15 19-54-17" src="https://github.com/user-attachments/assets/dcf29df1-9d0c-44f2-8697-497ea0b9bec3" />


### 2. Loading the .mp3 files
Loading the .mp3 files directly leads to crash due to consumption of all the ram.

Current loading process:
```python
import librosa

AUDIO_PATH = Path("/content/DEAM_audio/MEMD_audio")

audio_files = []
base = 15
clip_size = 5

for i in train_df["song_id"].unique():
  if i % 100:
    print(f"Loaded {i} files.")
  y, sr = librosa.load(AUDIO_PATH / f"{i}.mp3", sr=44100, mono=True)
  for j in range(6):
    start = int((base + clip_size*j) * sr)
    end   = int((base + clip_size*(j+1)) * sr)
    audio_files.append(y[start:end])
```

Storing raw data is not recommended, hence we directly store the mel spectograms on this disk.

Moving forward:
```python
def process_and_save_song(song_id):
    """Process and save to disk, return count of segments saved"""
    base = 15
    clip_size = 5
    saved_count = 0
    
    try:
        y, sr = librosa.load(AUDIO_PATH / f"{song_id}.mp3", sr=44100, mono=True)
        
        for j in range(6):
            start = int((base + clip_size*j) * sr)
            end = int((base + clip_size*(j+1)) * sr)
            clip = y[start:end]
            
            mel_spec = librosa.feature.melspectrogram(y=clip, sr=sr, n_mels=128, fmax=8000)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Save to disk
            filename = SPEC_DIR / f"song_{song_id}_seg_{j}.npy"
            np.save(filename, mel_spec_db)
            saved_count += 1
            
        return saved_count
        
    except Exception as e:
        print(f"Error processing song {song_id}: {e}")
        return 0
```


On running this core realisation hits of its runtime, its too slow.
To speed this up I added multiprocessing using `concurrent.futures`

### 3. PyTorch Dataset to the rescue

Loading data on demand and automated batch processing.


```python
  def __getitem__(self, idx):
    row = self.data.iloc[idx]
    seg = idx % 6
    arousal = row["arousal"]
    valence = row["valence"]
    mel_spec = np.load(SPEC_DIR / f"song_{int(row["song_id"])}_seg_{seg}.npy")

    return torch.FloatTensor(mel_spec), torch.FloatTensor([arousal, valence])
```



## CNN

```py
Architecture	Features	Kernel
Conv2d		
|
ReLU		  (1->16)		(3x3)
|
MaxPool

Conv2d
|
ReLU		  (16->32)	(3x3)
|
MaxPool

Flatten
|
Linear		  (32->2)

```
<img width="1021" height="470" alt="image" src="https://github.com/user-attachments/assets/dc6f4b81-6437-4681-a0fb-2fd1d406947e" />


The DEAM manual said arousal values are more noisy than valence, hence two different results.

```py
Mean baseline prediction: [0.12383987 0.07644205]
Baseline MSE: 0.06585912
Model MSE: 0.036714073
```

### Loss: MSE
Due to continous real values of arousal and valence
```py
valence ≈ -0.55 to +0.74
arousal ≈ -0.67 to +0.71
```
### Optimizer: Adam
For handling noisy gradients and moderate dataset.

# Generator
## Note Creation
Apparently to apply gradient descent all of the functions in the chain needs to be differentiable, i.e using a library to generate notes is not applicable, hence I am going to write the note generation code again in pytorch this time instead of numpy(PyMusic-Instrument).

## Methodology
Create an array of notes of a specific scale such as 5.
`fifth = [63, 61, 59, 57, 56, 54, 52]*7`
We can control the parameters such as duration, loudness and each note's onset following the DDSP algorithm.
`waveform = A(t) * sin(2pift)`
`A(t)` is the envelope function.
### Instrument: Piano
Craft relevant ADSR for piano.
Attack Decay Sustain Release.

* * *

## Ablation A: Direct Parameterization

```python
# Trainable parameters 
durations_ = torch.ones(len(fifth), requires_grad=True, device=device)
amplitudes_ = torch.ones(len(fifth), requires_grad=True, device=device)
```

### Problem:

We get this error when directly parameterizing duration. This likely happend due to one of the ADSR parameters reaching inf, due to the denominator tending to zero i.e `duration = A + D +R`.  
When one inf, backprop produced NaN parameters.

```python
Traceback (most recent call last):
  File "/home/hp/PycharmProjects/Fourier-s-Orchestra/main.py", line 143, in <module>
    audio = generate_note(fifth, durations_, amplitudes_)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hp/PycharmProjects/Fourier-s-Orchestra/main.py", line 89, in generate_note
    num_samples = int(44100 * duration.item())
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: cannot convert float NaN to integer
```

* * *

## Ablation B: Fixed-Canvas Temporal Masking

### Why switch to temporal masking (DDSP)?

Using method 1 to directly paramterize duration led variable tensor dimensions, which violated the fixed assumptions of backpropagation. Instead now we have desgined a 5 second canvas, with each note having an onset.  
Note onsets are initialised using `linspace` such that the time domain is evenly filled by the notes.

```python
# Trainable parameters
note_onsets = torch.linspace(0, 4, len(fifth), device=device, requires_grad=True)
note_durations = torch.full((len(fifth),), 0.5, device=device, requires_grad=True)
amplitudes = torch.full((len(fifth),), 0.7, device=device, requires_grad=True)
```

Clear distinction: `durations_` controlled how long the note existed v/s `note_durations` controls how long the note is audible.

### Problem:

Arousal and valence value are not independently controllable.

```python
# Loss
loss = torch.dist(pred_emotion, target)
```

Clearly using euclidean loss, makes the system closer to diagonally even if the valence(or arousal) drifts off too far from the target values.

Amplitude, Note density and short durations all affect arousal more, hence in few of our test runs we observed our generator only chasing arousal target value and the valence being handled according to arousal.

* * *

## Ablation C: ADSR Parameterization

Arousal is overall energy and valence is how this energy evolves.

Now parameterizing ADSR has introduced new degrees of freedom that primarily modulate the temporal distribution of energy rather than its magnitude. It enables the "partial" decoupling of valence from arousal values.

```python
A_ = torch.full((len(fifth), ), 0.01, device=device, requires_grad=True)
D_ = torch.full((len(fifth), ), 0.15, device=device, requires_grad=True)
R_ = torch.full((len(fifth), ), 0.7, device=device, requires_grad=True)
S_ = torch.full((len(fifth), ), 0., device=device, requires_grad=True)
```

## Problem:

### AdaptiveAvgPool2D Destroys Temporal Information

```python
nn.AdaptiveAvgPool2d((1, 1))
```

What this does it averages all the time frames, so something like:

> Sharp Attack -> Smooth Decay == Smooth Attack -> Sharp Decay

As long as the total energy distribution is similar.  
Down the line, the model was changed to preserve the time demension

```python
nn.AdaptiveAvgPool2d((1, None))
```

This is our "model_2.pt", although the model was learning its MSE was 0.04 i.e larger than the previous model, which encouraged us to research some other solution and abandon "model_2.pt".

* * *

## Ablation D: Sequential multi-objective optimization

Euclidean distance loss caused the optimizer to chase one dimension  
while ignoring the other, as arousal and valence have different sensitivities  
to musical parameters.

- Stage 1: Optimize arousal only using timing and amplitude parameters
    
    - ```python
         optimizer = torch.optim.Adam([note_onsets_, note_durations_, amplitudes_], lr=0.05)
         arousal_loss = (10* (pred_emotion[0, 0] - target[0, 0])) ** 2
         loss = arousal_loss
        ```
        
- Stage 2: Introduce valence optimization with ADSR parameters
    
    - ```python
        optimizer = torch.optim.Adam([note_onsets_, A_shared, D_shared, S_shared, R_shared], lr=0.03)
        arousal_loss = (10*(pred_emotion[0, 0] - target[0, 0])) ** 2
        valence_loss = (10*(pred_emotion[0, 1] - target[0, 1])) ** 2
        loss = valence_loss*2.5 + arousal_loss
        ```
        
- Stage 3: Joint fine tuning with equal weights and adaptive learning rate
    
    - ```python
        optimizer = torch.optim.Adam([note_onsets_, note_durations_, amplitudes_,
                               A_shared, D_shared, S_shared, R_shared], lr=0.1)
         arousal_loss = (10*(pred_emotion[0, 0] - target[0, 0])) ** 2
         valence_loss = (10*(pred_emotion[0, 1] - target[0, 1])) ** 2
         loss = arousal_loss + valence_loss
        ```
        

`torch.optim.lr_scheduler.ReduceLROnPlateau` was used to handle the learning rate in the 3rd step, it is done for quick minimization of loss and then lower learning rate to escape local minimas.

We changed the ADSR (192 parameters in total!!!) to 48 parameters.  
192 was a lot of gradient descent on a short 5 second clip.

```python
A_shared = torch.tensor([0.01], device=device, requires_grad=True)
D_shared = torch.tensor([0.15], device=device, requires_grad=True)
S_shared = torch.tensor([0.7], device=device, requires_grad=True)
R_shared = torch.tensor([0.3], device=device, requires_grad=True)
```

Till now our "model_1.pt" was not giving satisfactory result, so we decided to switch to a new model with batch processing to enhance the results i.e "model_3.pt"

* * *

# Proposed methodology

## Classifier

### Dataset: DEAM

Preprocessing to split 45 seconds clips to 5second clips starting from the 15 second mark.

### Input Representation

```python
mel_spec = librosa.feature.melspectrogram(y=clip, sr=sr, n_mels=128, fmax=8000, hop_length=512, n_fft=2048, win_length=2048,)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
```

### Train and Validation Split

```python
from sklearn.model_selection import train_test_split

unique_songs = train_df["song_id"].unique()
train_songs, val_songs = train_test_split(
    unique_songs,
    test_size=0.2,
    random_state=42
)

train_data = train_df[train_df["song_id"].isin(train_songs)]
val_data = train_df[train_df["song_id"].isin(val_songs)]
```

### CNN

```
Architecture                Features        Kernel / Params
----------------------------------------------------------------
Conv2d
|
BatchNorm2d
|
ReLU                        (1 → 16)        (3 × 3, padding=1)
|
MaxPool2d                   ↓               (2 × 2)

Conv2d
|
BatchNorm2d
|
ReLU                        (16 → 32)       (3 × 3, padding=1)
|
MaxPool2d                   ↓               (2 × 2)

Conv2d
|
BatchNorm2d
|
ReLU                        (32 → 64)       (3 × 3, padding=1)

AdaptiveAvgPool2d           ↓               (1 × 1)

Flatten                     (B, 64, 1, 1 → B, 64)

Linear                      (64 → 2)
----------------------------------------------------------------
Output: (B, 2)
```

- Loss: MSE
- Optimizer: Adam
- Learning rate: 1e-3
- Epochs: 30

## Generator

### Problem Foundation

1.  Generator is optimized to match target arousal and valence values.
2.  The trained CNN is frozen and used as emotional oracle.
3.  Optimization is differentiable.

We define emotion conditioned audio generation as an inverse optimization problem.

$$
\text{Given target emotion } \mathbf{e}_t = (a_t, v_t),
\text{ find synthesis parameters } \boldsymbol{\theta}
\text{ such that}
$$

$$
f_{\text{emo}}(g(\boldsymbol{\theta})) \approx \mathbf{e}_t
$$

### Differentiable Synthesis Model

$$
x(t) = \sum_{i=1}^{N}
\sigma\!\left(\alpha (t - o_i)\right)
\sigma\!\left(\alpha (o_i + d_i - t)\right)
\, a_i\,E(t - o_i)\,
\sin(2\pi f_i t)
$$

$$
E(t) =
\begin{cases}
\frac{t}{A}, & 0 \le t < A \\[6pt]
1 - (1 - S)\frac{t - A}{D}, & A \le t < A + D \\[6pt]
S \exp\!\left(-\beta \frac{t - (A + D)}{d - A - D - R}\right), & A + D \le t < d - R \\[6pt]
S \exp\!\left(-\beta \frac{t - (d - R)}{R}\right), & d - R \le t < d
\end{cases}
$$

where β>0 controls exponential decay smoothness.

Let the following notation be used throughout the differentiable synthesis model:

$t \in [0, T]$ - continuous time variable over a fixed synthesis duration $T$.
$x(t)$ - synthesized audio waveform in the time domain.
$x_i(t)$ - signal corresponding to the $i$-th note component.
$f_i$ - fundamental frequency (Hz) of the $i$-th note, computed as
$o_i$ - onset time (seconds) of the $i$-th note.
$d_i$ - duration (seconds) of the $i$-th note.
$a_i$ - amplitude scaling factor of the $i$-th note.
$m_i(t)$ - differentiable temporal mask controlling note activation.
$\sigma(\cdot)$ - sigmoid activation function.
$\alpha$ - sharpness parameter controlling onset/offset transitions.
$E(t)$ - ADSR envelope function.
$A$ - attack time parameter.
$D$ - decay time parameter.
$S$ - sustain level parameter.
$R$ - release time parameter.
$\boldsymbol{\theta}$ - set of all trainable synthesis parameters.

### Parameterization of Musical Events
1. Note sequence initialisation
	- Fixed pitch sequence `octa = [60.0, 62.0, 64.0, 65.0, 67.0, 51.0, 49.0, 47.0, 45.0, 44.0, 42.0, 40.0]*4`
	- Randomized ordering `fifth[torch.randperm(fifth.size(0))]`
	- Evenly spaced initial onsets `note_onsets_ = torch.linspace(0, 3.5, len(fifth), device=device, requires_grad=True)`
  
2. Trainable Parameters

	- Timing parameters - (onsets, durations)
	- Amplitude parameters - (note loudness)
	- Envelope parameters (ADSR)- (attack, decay, sustain, release)

### Parameter Constraints and Reparameterization
```python
onsets_positive = torch.sigmoid(note_onsets_) * 4.0
durations_positive = softplus(note_durations_) + 0.1
amplitudes_positive = sigmoid(amplitudes_)

A_val = sigmoid(A_shared) * 0.099 + 0.001
D_val = sigmoid(D_shared) * 0.49 + 0.01
S_val = sigmoid(S_shared) * 0.6 + 0.3
R_val = sigmoid(R_shared) * 0.95 + 0.05
```

All trainable synthesis parameters are reparameterized using sigmoid or softplus transformations to enforce physically meaningful ranges (for example: positive durations, bounded amplitudes) and to prevent numerical instability during optimization such as geting `nan(s)`

### Fixed-Canvas temporal masking

Direct parameterization of note durations leads to variable length signals, which violates the fixed assumptions of back propagation
To ensure differentiability and stable back propagation synthesis in performed on a fixed canvas of 5 seconds. This fixed canvas formulation is inspired from DDSP-style synthesis, where temporal strcuture is controlled through continuous masks instead of discrete boundaries.

### Emotion Feedback loop
Our generated audio waveforms are transformed into mel spectograms in logarithmic scale using the same preprocessing pipeline. These spectograms are then passed to the pretrained frozen emotion regression CNN.
The predicted arousal and valence values serve as feedback signals, the resulting loss is backpropagated through the synthesis model to recompute the synthesis parameters. This is establishes our differentiable feedback loop.

### Loss Function

Loss function is scaled by magnitude, as the general values of arousal and valence are already less than 1.
Both arousal and valence loss are MSE based, while the arousal is most sensitive to overall signal energy, note density and amplitude, valence is administered strongly by temporal evolution of energy such that it relies heavily on ADSR parameters than on amplitude alone.


### Sequential Multi-Objective Optimization

- Stage I: Arousal Optimization

In the first stage, only timing and amplitude parameters are optimized to match the target arousal value, while valence is ignored. This allows the model to establish an appropriate global energy level before introducing more sensitive temporal controls.

- Stage II: Valence Introduction

In the second stage, ADSR envelope parameters are introduced and valence loss is added with a higher weight. This encourages the model to adjust the temporal distribution of energy while maintaining the previously optimized arousal level.

- Stage III: Joint Fine Tuning

Finally, all synthesis parameters are jointly optimized using equal weighting for arousal and valence. An adaptive learning-rate scheduler is employed to refine convergence and mitigate local minima.

### Optimization details

- Optimizer: Adam
- Gradient clipping
- Learning rate scheduling using `ReduceLROnPlateau`
- Single sample optimization
- Early stopping criteria

### Ablation Summary
Several design choices were evaluated through ablation.
- Direct duration parameterization resulted in mumerical instability.
- Fixed canvas synthesis without ADSR parameters failed to decouple valence and arousal values.
- Global temporal pooling resulted in compromised envelope cues.
- Join optimization was dominated by arousal, which gave motivation for sequential optimization.

## Limitations
No perceptual listening studies are conducted, and emotion control is assessed only through model predictions. Human evaluation is left for future work.
