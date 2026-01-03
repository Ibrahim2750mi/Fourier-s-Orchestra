import torch
import torchaudio
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasicCNN(nn.Module):
  def __init__(self):
    super().__init__()

    self.features = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.AdaptiveAvgPool2d((1, 1))
    )

    self.regressor = nn.Linear(64, 2)

  def forward(self, x):
    x = self.features(x)      # (B, 64, 1, 1)
    x = x.flatten(1)          # (B, 64)
    return self.regressor(x)  # (B, 2)

def generate_note_ddsp(keys, note_onsets, note_durations, amplitudes, env_elems,
                       total_duration=5.0, sample_rate=44100):
    """
    DDSP-style generation: all notes exist in a fixed time grid.

    :param keys: Fixed MIDI notes [num_notes]
    :param note_onsets: When each note starts (0 to total_duration) [num_notes]
    :param note_durations: How long each note lasts [num_notes]
    :param amplitudes: Volume of each note [num_notes]
    :param total_duration: Total audio length in seconds
    :return: Fixed-length audio waveform
    """

    num_samples = int(sample_rate * total_duration)
    t = torch.linspace(0, total_duration, num_samples, device=device)
    audio = torch.zeros(num_samples, device=device)

    for i in range(len(keys)):
        # Convert MIDI to frequency
        key_hz = 440.0 * torch.pow(2.0, (keys[i] - 69) / 12.0)

        onset = note_onsets[i]
        duration = note_durations[i]
        amplitude = amplitudes[i]

        note_start = onset
        note_end = onset + duration

        # Smooth onset (fade in over 0.01s)
        onset_window = torch.sigmoid((t - note_start) * 100)

        # Smooth offset (fade out over 0.01s)
        offset_window = torch.sigmoid(-(t - note_end) * 100)

        note_window = onset_window * offset_window

        # ADSR envelope (relative to note start)
        t_relative = t - note_start
        envelope = piano_envelope_ddsp(t_relative, duration, amplitude, env_elems[0], env_elems[1],
                                       env_elems[2], env_elems[3])

        # Generate sine wave
        phase = 2 * torch.pi * key_hz * t_relative
        wave = torch.sin(phase)

        note_signal = wave * envelope * note_window

        audio = audio + note_signal

    return audio


def piano_envelope_ddsp(t_relative, duration, amplitude, A=0.01, D=0.15, S=0.7, R=0.3):
    """
    ADSR envelope for DDSP approach.
    Works with relative time (from note onset).
    """

    # Clamp negative times to 0 (before note starts)
    t_relative = torch.clamp(t_relative, min=0)

    # Attack
    attack = torch.clamp(t_relative / A, max=1.0)

    # Decay
    decay_t = torch.clamp((t_relative - A) / D, min=0, max=1.0)
    decay = 1 - (1 - S) * decay_t

    # Sustain (exponential fade)
    sustain_t = torch.clamp((t_relative - A - D) / (duration - A - D - R + 1e-6), min=0, max=1.0)
    sustain = S * torch.exp(-1.5 * sustain_t)

    # Release
    release_start = duration - R
    release_t = torch.clamp((t_relative - release_start) / R, min=0, max=1.0)
    release = S * torch.exp(torch.tensor(-1.5, device=t_relative.device)) * (1 - release_t)

    # Combine phases using soft blending
    w_attack = torch.sigmoid(100 * (A - t_relative))
    w_decay = torch.sigmoid(100 * (A + D - t_relative)) * (1 - w_attack)
    w_sustain = torch.sigmoid(100 * (release_start - t_relative)) * (1 - w_decay)
    w_release = 1 - w_sustain

    envelope = (
            w_attack * attack +
            w_decay * decay +
            w_sustain * sustain +
            w_release * release
    )

    return amplitude * envelope


model = BasicCNN().to(device)
state = torch.load("model_3.pt", map_location=device)
model.load_state_dict(state)
model.eval()

# Notes
octa = [60.0, 62.0, 64.0, 65.0, 67.0, 51.0, 49.0, 47.0, 45.0, 44.0, 42.0, 40.0]*4

fifth = torch.tensor(octa, device=device)
fifth[:] = fifth[torch.randperm(fifth.size(0))]


# Initialize for energetic sound
note_onsets_ = torch.linspace(0, 3.5, len(fifth), device=device, requires_grad=True)
note_durations_ = torch.full((len(fifth),), 0.3, device=device, requires_grad=True)
amplitudes_ = torch.full((len(fifth),), 0.8, device=device, requires_grad=True)

A_shared = torch.tensor([0.01], device=device, requires_grad=True)
D_shared = torch.tensor([0.15], device=device, requires_grad=True)
S_shared = torch.tensor([0.7], device=device, requires_grad=True)
R_shared = torch.tensor([0.3], device=device, requires_grad=True)

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=44100, n_mels=128, f_max=8000, n_fft=2048,win_length=512,
).to(device)
amp_to_db = torchaudio.transforms.AmplitudeToDB().to(device)

final_target_arousal = 0.7
final_target_valence = -0.5


print(f"\n1. Optimizing Arousal → {final_target_arousal}")

target = torch.tensor([[final_target_arousal, 0.0]], device=device)

optimizer = torch.optim.Adam([note_onsets_, note_durations_, amplitudes_], lr=0.05)

for step in range(300):
    optimizer.zero_grad()

    onsets_positive = torch.sigmoid(note_onsets_) * 4.0
    durations_positive = torch.nn.functional.softplus(note_durations_) + 0.1
    amplitudes_positive = torch.sigmoid(amplitudes_)

    A_val = torch.sigmoid(A_shared) * 0.099 + 0.001

    D_val = torch.sigmoid(D_shared) * 0.49 + 0.01
    S_val = torch.sigmoid(S_shared) * 0.6 + 0.3
    R_val = torch.sigmoid(R_shared) * 0.95 + 0.05

    onsets_sorted, sort_idx = torch.sort(onsets_positive)

    audio = generate_note_ddsp(
        fifth[sort_idx], onsets_sorted,
        durations_positive[sort_idx], amplitudes_positive[sort_idx],
        [A_val, D_val, S_val, R_val], total_duration=5.0
    )

    audio = torch.clamp(audio, min=-1.0, max=1.0)

    # Add small epsilon to avoid log(0) in spectrogram
    audio = audio + 1e-8

    spec = mel_transform(audio)
    spec_db = amp_to_db(spec)
    spec_db = spec_db.unsqueeze(0).unsqueeze(0)

    pred_emotion = model(spec_db)

    # Only optimize arousal
    arousal_loss = (10* (pred_emotion[0, 0] - target[0, 0])) ** 2
    loss = arousal_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_([note_onsets_, note_durations_, amplitudes_], max_norm=1.0)
    optimizer.step()

    if loss.item() < 1:
        print(f"\nConverged! Loss {loss.item()} < 1 at step {step}")
        break

    if step % 30 == 0:
        print(f"Step {step}: Arousal={pred_emotion[0, 0].item()} "
              f"(target={target[0, 0].item()}), Loss={loss.item()}")

print(f"\nComplete: Arousal={pred_emotion[0, 0].item()}, "
      f"Valence={pred_emotion[0, 1].item()}")

print(f"\n2. Valence → {final_target_valence}")

target = torch.tensor([[final_target_arousal, final_target_valence]], device=device)

# Now optimize ADSR too
optimizer = torch.optim.Adam([note_onsets_, A_shared, D_shared, S_shared, R_shared], lr=0.03)

for step in range(250):
    optimizer.zero_grad()

    onsets_positive = torch.sigmoid(note_onsets_) * 4.0
    durations_positive = torch.nn.functional.softplus(note_durations_) + 0.1
    amplitudes_positive = torch.sigmoid(amplitudes_)

    # Attack: 0.001 to 0.1 seconds (very fast to fast)
    A_val = torch.sigmoid(A_shared) * 0.099 + 0.001

    # Decay: 0.01 to 0.5 seconds
    D_val = torch.sigmoid(D_shared) * 0.49 + 0.01

    # Sustain: 0.3 to 0.9 (level, not time)
    S_val = torch.sigmoid(S_shared) * 0.6 + 0.3

    # Release: 0.05 to 1.0 seconds
    R_val = torch.sigmoid(R_shared) * 0.95 + 0.05

    onsets_sorted, sort_idx = torch.sort(onsets_positive)

    audio = generate_note_ddsp(
        fifth[sort_idx], onsets_sorted,
        durations_positive[sort_idx], amplitudes_positive[sort_idx],
        [A_val, D_val, S_val, R_val], total_duration=5.0
    )

    spec = mel_transform(audio)
    spec_db = amp_to_db(spec)
    spec_db = spec_db.unsqueeze(0).unsqueeze(0)

    pred_emotion = model(spec_db)

    # Weight valence more heavily now
    arousal_loss = (10*(pred_emotion[0, 0] - target[0, 0])) ** 2
    valence_loss = (10*(pred_emotion[0, 1] - target[0, 1])) ** 2
    loss = valence_loss  # Emphasize valence

    loss.backward()
    torch.nn.utils.clip_grad_norm_([note_onsets_, note_durations_, amplitudes_,
                                    A_shared, D_shared, S_shared, R_shared], max_norm=1.0)
    optimizer.step()

    if loss.item() < 1:
        print(f"\nConverged! Loss {loss.item()} < 1 at step {step}")
        break

    if step % 30 == 0:
        print(f"Step {step}: Arousal={pred_emotion[0, 0].item()}, "
              f"Valence={pred_emotion[0, 1].item()}, Loss={loss.item()}")

print(f"\nComplete: Arousal={pred_emotion[0, 0].item()}, "
      f"Valence={pred_emotion[0, 1].item()}")

print(f"\n3. Fine-tuning")

optimizer = torch.optim.Adam([note_onsets_, note_durations_, amplitudes_,
                              A_shared, D_shared, S_shared, R_shared], lr=0.1)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=30
)

best_loss = float('inf')

for step in range(200):
    optimizer.zero_grad()

    onsets_positive = torch.sigmoid(note_onsets_) * 4.0
    durations_positive = torch.nn.functional.softplus(note_durations_) + 0.1
    amplitudes_positive = torch.sigmoid(amplitudes_)

    # Attack: 0.001 to 0.1 seconds (very fast to fast)
    A_val = torch.sigmoid(A_shared) * 0.099 + 0.001

    # Decay: 0.01 to 0.5 seconds
    D_val = torch.sigmoid(D_shared) * 0.49 + 0.01

    # Sustain: 0.3 to 0.9 (level, not time)
    S_val = torch.sigmoid(S_shared) * 0.6 + 0.3

    # Release: 0.05 to 1.0 seconds
    R_val = torch.sigmoid(R_shared) * 0.95 + 0.05

    onsets_sorted, sort_idx = torch.sort(onsets_positive)

    audio = generate_note_ddsp(
        fifth[sort_idx], onsets_sorted,
        durations_positive[sort_idx], amplitudes_positive[sort_idx],
        [A_val, D_val, S_val, R_val], total_duration=5.0
    )

    spec = mel_transform(audio)
    spec_db = amp_to_db(spec)
    spec_db = spec_db.unsqueeze(0).unsqueeze(0)

    pred_emotion = model(spec_db)

    # Equal weights
    arousal_loss = (10*(pred_emotion[0, 0] - target[0, 0])) ** 2
    valence_loss = (10*(pred_emotion[0, 1] - target[0, 1])) ** 2
    loss = arousal_loss + valence_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_([note_onsets_, note_durations_, amplitudes_,
                                    A_shared, D_shared, S_shared, R_shared], max_norm=1.0)
    optimizer.step()
    scheduler.step(loss)

    if loss.item() < best_loss:
        best_loss = loss.item()

    if loss.item() < 10:
        print(f"\nConverged! Loss {loss.item()} < 10 at step {step}")
        break

    if step % 40 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Step {step}: Arousal={pred_emotion[0, 0].item()}, "
              f"Valence={pred_emotion[0, 1].item()}, "
              f"Loss={loss.item()}, LR={current_lr}")

print(f"\nStage 3 Complete: Arousal={pred_emotion[0, 0].item()}, "
      f"Valence={pred_emotion[0, 1].item()}")

print("OPTIMIZATION COMPLETE")
print(f"Target:     Arousal={final_target_arousal}, Valence={final_target_valence}")
print(f"Achieved:   Arousal={pred_emotion[0, 0].item()}, Valence={pred_emotion[0, 1].item()}")
print(f"Final Loss: {loss.item()}")
print(f"Best Loss:  {best_loss}")

# Save audio
audio = audio.detach().cpu()
audio = audio / (audio.abs().max() + 1e-6)
audio = audio.unsqueeze(0)
torchaudio.save(f"output_arousal{final_target_arousal}_valence{final_target_valence}.wav",
                audio, 44100)
print(f"\nSaved: output_arousal{final_target_arousal}_valence{final_target_valence}.wav")