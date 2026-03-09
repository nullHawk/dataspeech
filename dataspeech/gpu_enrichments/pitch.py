import torch
import torch.nn.functional as F
from ._torbi_compat import ensure_torbi
ensure_torbi()
import penn


# Here we'll use a 10 millisecond hopsize
hopsize = .01

# Provide a sensible frequency range given your domain and model
fmin = 30.
fmax = 1000.

# Select a checkpoint to use for inference. Selecting None will
# download and use FCNF0++ pretrained on MDB-stem-synth and PTDB
checkpoint = None

# Centers frames at hopsize / 2, 3 * hopsize / 2, 5 * hopsize / 2, ...
center = 'half-hop'

# (Optional) Linearly interpolate unvoiced regions below periodicity threshold
interp_unvoiced_at = .065


def pitch_apply(batch, rank=None, audio_column_name="audio", output_column_name="utterance_pitch", penn_batch_size=4096):
    gpu = (rank or 0) % torch.cuda.device_count() if torch.cuda.device_count() > 0 else rank

    if isinstance(batch[audio_column_name], list):
        samples = batch[audio_column_name]

        # Pad all audios to max length and stack into single tensor
        arrays = [torch.tensor(s["array"][None, :]).float() for s in samples]
        max_len = max(a.shape[-1] for a in arrays)
        lengths = [a.shape[-1] for a in arrays]
        padded = torch.stack([
            F.pad(a, (0, max_len - a.shape[-1])) for a in arrays
        ]).squeeze(1)  # (N, 1, max_len)

        sr = samples[0]["sampling_rate"]

        with torch.no_grad():
            pitch, periodicity = penn.from_audio(
                padded, sr,
                hopsize=hopsize, fmin=fmin, fmax=fmax,
                checkpoint=checkpoint, batch_size=penn_batch_size,
                center=center, interp_unvoiced_at=interp_unvoiced_at,
                gpu=gpu,
            )

        # Mask out padded regions per sample
        utterance_pitch_mean = []
        utterance_pitch_std = []
        for i, orig_len in enumerate(lengths):
            real_frames = int(orig_len / (sr * hopsize))
            p = pitch[i, :real_frames]
            utterance_pitch_mean.append(p.mean().cpu())
            utterance_pitch_std.append(p.std().cpu())

        batch[f"{output_column_name}_mean"] = utterance_pitch_mean
        batch[f"{output_column_name}_std"] = utterance_pitch_std
    else:
        sample = batch[audio_column_name]
        with torch.no_grad():
            pitch, periodicity = penn.from_audio(
                torch.tensor(sample["array"][None, :]).float(),
                sample["sampling_rate"],
                hopsize=hopsize, fmin=fmin, fmax=fmax,
                checkpoint=checkpoint, batch_size=penn_batch_size,
                center=center, interp_unvoiced_at=interp_unvoiced_at,
                gpu=gpu,
            )
        batch[f"{output_column_name}_mean"] = pitch.mean().cpu()
        batch[f"{output_column_name}_std"] = pitch.std().cpu()

    return batch
