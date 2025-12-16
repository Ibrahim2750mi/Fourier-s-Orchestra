# Fourier-s-Orchestra
Submission for BYOP 2025


## Figuring out how to use data
###  1. Dynamic annotations v/s static annotations:
Dreferring dynamic so we have more training data, starting from 15s, with 5second clip size.

Initital df structure.
![d6fb5fe64e614e3d230cccc190384f44.png](:/1905335258c0444cbb9f4502689e89e9)

Final df structure:
![5c14b900993006a5a0ba23bfe30508a9.png](:/a2cfb43a346f4e929382f135051475a5)

### 2. Loading the .mp3 files
Loading the .mp3 files directly leads to crash due to consumption of all the ram.

Current loading process:
```
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


Initialy the spec range was from [-80, 0], converted it to [0,1] for activation function.
