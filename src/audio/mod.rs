use std::io::Cursor;

use anyhow::{Context, Result};
use hound::{SampleFormat, WavSpec, WavWriter};

const DEFAULT_PEAK_TARGET: f32 = 0.97;

pub fn normalize_peak(samples: &mut [f32]) {
    normalize_peak_to(samples, DEFAULT_PEAK_TARGET);
}

pub fn normalize_peak_to(samples: &mut [f32], target: f32) {
    if samples.is_empty() {
        return;
    }
    let peak = samples
        .iter()
        .fold(0.0_f32, |max, &value| max.max(value.abs()));
    if peak > 0.0 {
        let gain = target / peak;
        for sample in samples.iter_mut() {
            *sample *= gain;
        }
    }
}

pub fn pcm_to_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let payload_bytes = samples.len().saturating_mul(2);
    let mut cursor = Cursor::new(Vec::with_capacity(payload_bytes.saturating_add(128)));
    {
        let spec = WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer =
            WavWriter::new(&mut cursor, spec).context("failed to initialise WAV writer")?;
        for sample in samples {
            let scaled = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            writer
                .write_sample(scaled)
                .context("failed to write WAV sample")?;
        }
        writer.finalize().context("failed to finalise WAV writer")?;
    }
    Ok(cursor.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn normalize_peak_scales_to_target() {
        let mut samples = vec![0.5_f32, -0.25_f32];
        normalize_peak_to(&mut samples, 0.25);
        assert!((samples[0] - 0.25).abs() < 1e-6);
        assert!((samples[1] + 0.125).abs() < 1e-6);
    }

    #[test]
    fn pcm_to_wav_roundtrip_preserves_length() {
        let samples = vec![0.0_f32, 0.5_f32, -0.5_f32];
        let wav = pcm_to_wav(&samples, 22050).expect("wav encoding");
        let mut reader = hound::WavReader::new(Cursor::new(wav)).expect("wav decoding failed");
        let decoded: Vec<i16> = reader
            .samples::<i16>()
            .map(|s| s.expect("sample read"))
            .collect();
        assert_eq!(decoded.len(), samples.len());
        assert_eq!(reader.spec().sample_rate, 22050);
    }
}
