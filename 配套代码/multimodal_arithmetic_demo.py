"""Chapter 83: representation arithmetic only, not a trained image/audio model."""
import json
import math


def patch_positions(height, width, patch):
    if height % patch or width % patch:
        raise ValueError("teaching example requires exactly divisible shape")
    return [(row, col) for row in range(height // patch) for col in range(width // patch)]


def main():
    positions = patch_positions(224, 224, 16)
    assert len(positions) == 196 and positions[14] == (1, 0)
    raw_patch_size = 16 * 16 * 3
    projection_parameters = raw_patch_size * 768
    assert raw_patch_size == 768 and projection_parameters == 589824
    sample_count = 16000 * 30
    assert sample_count == 480000
    one_second_frames = (16000 - 400) // 160 + 1
    assert one_second_frames == 98
    positive_probability = math.exp(2) / (math.exp(2) + 1)
    loss = -math.log(positive_probability)
    assert abs(positive_probability - .8807970779778824) < 1e-12
    assert abs(loss - .12692801104297263) < 1e-12
    assert (2 + 1 + 1) / 20 == .2
    print(json.dumps({"passed": True, "patches": len(positions), "patch_projection_parameters": projection_parameters,
                      "audio_samples": sample_count, "frames_without_padding": one_second_frames,
                      "contrastive_positive_probability": positive_probability, "contrastive_loss": loss,
                      "scope": "synthetic arithmetic; no CLIP or Whisper inference"}))


if __name__ == "__main__":
    main()
