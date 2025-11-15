use std::io::Cursor;
#[cfg(feature = "mp3")]
use std::ptr;

use anyhow::{Context, Result, anyhow, bail};
use hound::{SampleFormat, WavSpec, WavWriter};
#[cfg(feature = "mp3")]
use libc::c_int;

const DEFAULT_PEAK_TARGET: f32 = 0.97;
#[cfg(feature = "mp3")]
const DEFAULT_MP3_BITRATE: c_int = 192;
#[cfg(feature = "mp3")]
const MP3_PADDING: usize = 7200;

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

#[cfg(feature = "mp3")]
pub fn pcm_to_mp3(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let mut encoder = LameEncoder::new(sample_rate, 1)?;
    encoder.encode(samples)
}

#[cfg(not(feature = "mp3"))]
pub fn pcm_to_mp3(_samples: &[f32], _sample_rate: u32) -> Result<Vec<u8>> {
    bail!("MP3 output is disabled (rebuild with `--features mp3` and install libmp3lame)");
}

#[cfg(feature = "mp3")]
struct LameEncoder {
    inner: *mut lame_global_flags,
}

#[cfg(feature = "mp3")]
impl LameEncoder {
    fn new(sample_rate: u32, channels: c_int) -> Result<Self> {
        unsafe {
            let handle = lame_init();
            if handle.is_null() {
                bail!("failed to initialise libmp3lame encoder");
            }
            let mut encoder = Self { inner: handle };
            encoder.configure(sample_rate, channels)?;
            Ok(encoder)
        }
    }

    fn configure(&mut self, sample_rate: u32, channels: c_int) -> Result<()> {
        unsafe {
            let sr = sample_rate
                .try_into()
                .map_err(|_| anyhow!("sample rate {sample_rate} too large"))?;
            ensure_success(
                lame_set_in_samplerate(self.inner, sr),
                "lame_set_in_samplerate",
            )?;
            ensure_success(
                lame_set_out_samplerate(self.inner, sr),
                "lame_set_out_samplerate",
            )?;
            ensure_success(
                lame_set_num_channels(self.inner, channels),
                "lame_set_num_channels",
            )?;
            ensure_success(
                lame_set_brate(self.inner, DEFAULT_MP3_BITRATE),
                "lame_set_brate",
            )?;
            ensure_success(lame_set_quality(self.inner, 2), "lame_set_quality")?;
            ensure_success(lame_init_params(self.inner), "lame_init_params")
        }
    }

    fn encode(&mut self, samples: &[f32]) -> Result<Vec<u8>> {
        let buffer_size = estimate_mp3_buffer(samples.len());
        let mut mp3 = Vec::with_capacity(buffer_size);
        let mut scratch = vec![0u8; buffer_size];

        let sample_len: c_int = samples
            .len()
            .try_into()
            .map_err(|_| anyhow!("audio buffer too large for MP3 encoder"))?;
        let written = unsafe {
            lame_encode_buffer_ieee_float(
                self.inner,
                samples.as_ptr(),
                ptr::null(),
                sample_len,
                scratch.as_mut_ptr(),
                scratch.len() as c_int,
            )
        };
        ensure_success(written, "lame_encode_buffer_ieee_float")?;
        mp3.extend_from_slice(&scratch[..written as usize]);

        let flushed =
            unsafe { lame_encode_flush(self.inner, scratch.as_mut_ptr(), scratch.len() as c_int) };
        ensure_success(flushed, "lame_encode_flush")?;
        mp3.extend_from_slice(&scratch[..flushed as usize]);
        Ok(mp3)
    }
}

#[cfg(feature = "mp3")]
impl Drop for LameEncoder {
    fn drop(&mut self) {
        unsafe {
            if !self.inner.is_null() {
                lame_close(self.inner);
                self.inner = ptr::null_mut();
            }
        }
    }
}

#[cfg(feature = "mp3")]
fn estimate_mp3_buffer(samples: usize) -> usize {
    (((samples as f64 * 1.25).ceil() as usize) + MP3_PADDING).max(MP3_PADDING)
}

#[cfg(feature = "mp3")]
fn ensure_success(code: c_int, func: &str) -> Result<()> {
    if code < 0 {
        bail!("{func} failed with code {code}");
    }
    Ok(())
}

#[cfg(feature = "mp3")]
#[repr(C)]
struct lame_global_flags {
    _private: [u8; 0],
}

#[cfg(feature = "mp3")]
#[link(name = "mp3lame")]
unsafe extern "C" {
    fn lame_init() -> *mut lame_global_flags;
    fn lame_close(gfp: *mut lame_global_flags) -> c_int;
    fn lame_set_in_samplerate(gfp: *mut lame_global_flags, value: c_int) -> c_int;
    fn lame_set_out_samplerate(gfp: *mut lame_global_flags, value: c_int) -> c_int;
    fn lame_set_num_channels(gfp: *mut lame_global_flags, value: c_int) -> c_int;
    fn lame_set_brate(gfp: *mut lame_global_flags, value: c_int) -> c_int;
    fn lame_set_quality(gfp: *mut lame_global_flags, value: c_int) -> c_int;
    fn lame_init_params(gfp: *mut lame_global_flags) -> c_int;
    fn lame_encode_buffer_ieee_float(
        gfp: *mut lame_global_flags,
        pcm_l: *const f32,
        pcm_r: *const f32,
        nsamples: c_int,
        mp3buf: *mut u8,
        mp3buf_size: c_int,
    ) -> c_int;
    fn lame_encode_flush(gfp: *mut lame_global_flags, mp3buf: *mut u8, size: c_int) -> c_int;
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

    #[cfg(feature = "mp3")]
    #[test]
    fn pcm_to_mp3_produces_bytes() {
        let samples = vec![0.0_f32; 22050];
        let mp3 = pcm_to_mp3(&samples, 22050).expect("mp3 encoding");
        let frame_sync = mp3
            .get(0)
            .copied()
            .zip(mp3.get(1).copied())
            .map(|(b0, b1)| b0 == 0xFF && (b1 & 0xE0) == 0xE0)
            .unwrap_or(false);
        assert!(
            mp3.starts_with(b"ID3") || frame_sync,
            "mp3 header not detected"
        );
    }
}
