use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use jieba_rs::Jieba;
use once_cell::sync::Lazy;
use pinyin::{Pinyin, ToPinyin};

use crate::nlp::PUNCTUATIONS;

use super::tone_sandhi::{TONE_SANDHI, ToneSandhi};
use crate::nlp::english;

static PINYIN_TO_SYMBOL_MAP: Lazy<HashMap<String, Vec<String>>> = Lazy::new(|| {
    let data = include_str!("../../../resources/opencpop-strict.txt");
    data.lines()
        .filter_map(|line| {
            let mut parts = line.split('\t');
            let key = parts.next()?.trim().to_string();
            let values = parts.next().map(|rest| {
                rest.split_whitespace()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
            })?;
            Some((key, values))
        })
        .collect()
});

static JIEBA: Lazy<Jieba> = Lazy::new(Jieba::new);

const PINYIN_INITIALS: [&str; 23] = [
    "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r",
    "z", "c", "s", "y", "w",
];

pub fn g2p(text: &str) -> Result<(Vec<String>, Vec<i32>, Vec<usize>)> {
    let tone_modifier: &ToneSandhi = &TONE_SANDHI;
    let mut phones = Vec::new();
    let mut tones = Vec::new();
    let mut word2ph = Vec::new();

    for sentence in split_sentences(text) {
        if sentence.trim().is_empty() {
            continue;
        }
        let (seg_phones, seg_tones, seg_word2ph) =
            process_sentence(sentence.trim(), tone_modifier)?;
        phones.extend(seg_phones);
        tones.extend(seg_tones);
        word2ph.extend(seg_word2ph);
    }

    phones.insert(0, "_".into());
    phones.push("_".into());
    tones.insert(0, 0);
    tones.push(0);
    word2ph.insert(0, 1);
    word2ph.push(1);

    let expected_len = text.chars().count() + 2;
    if word2ph.len() < expected_len {
        let deficit = expected_len - word2ph.len();
        word2ph.extend(std::iter::repeat(0).take(deficit));
    } else if word2ph.len() > expected_len {
        word2ph.truncate(expected_len);
    }
    if let Some(first) = word2ph.first_mut() {
        *first = 1;
    }
    if let Some(last) = word2ph.last_mut() {
        *last = 1;
    }

    let chars: Vec<char> = text.chars().collect();
    for (idx, &ch) in chars.iter().enumerate() {
        if ch.is_ascii_whitespace() {
            let pos = idx + 1;
            if pos < word2ph.len().saturating_sub(1) {
                word2ph[pos] = 0;
            }
        }
    }

    let sum = word2ph.iter().map(|&v| v as isize).sum::<isize>();
    let target = phones.len() as isize;
    if sum != target {
        if sum < target {
            let mut remaining = target - sum;
            for (idx, &ch) in chars.iter().enumerate() {
                if remaining == 0 {
                    break;
                }
                let pos = idx + 1;
                if pos >= word2ph.len().saturating_sub(1) {
                    continue;
                }
                if ch.is_ascii_whitespace() {
                    continue;
                }
                word2ph[pos] += 1;
                remaining -= 1;
            }
            if remaining > 0 {
                if let Some(last) = word2ph.last_mut() {
                    *last += remaining as usize;
                }
            }
        } else {
            let mut remaining = sum - target;
            for (idx, &ch) in chars.iter().enumerate().rev() {
                if remaining == 0 {
                    break;
                }
                let pos = idx + 1;
                if pos >= word2ph.len().saturating_sub(1) {
                    continue;
                }
                if word2ph[pos] == 0 {
                    continue;
                }
                if ch.is_ascii_whitespace() {
                    continue;
                }
                let reduce = remaining.min(word2ph[pos] as isize);
                word2ph[pos] -= reduce as usize;
                remaining -= reduce;
            }
            if remaining > 0 {
                if let Some(last) = word2ph.last_mut() {
                    let reduce = remaining.min(*last as isize);
                    *last -= reduce as usize;
                }
            }
        }
    }

    if let Some(first) = word2ph.first_mut() {
        *first = 1;
    }
    if let Some(last) = word2ph.last_mut() {
        if *last == 0 {
            *last = 1;
        }
    }

    Ok((phones, tones, word2ph))
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if is_punctuation_char(ch) {
            if !current.trim().is_empty() {
                sentences.push(current.clone());
            }
            current.clear();
        }
    }
    if !current.trim().is_empty() {
        sentences.push(current);
    }
    sentences
}

fn process_sentence(
    sentence: &str,
    tone_modifier: &ToneSandhi,
) -> Result<(Vec<String>, Vec<i32>, Vec<usize>)> {
    let mut tagged: Vec<(String, String)> = JIEBA
        .tag(sentence, true)
        .into_iter()
        .map(|t| (t.word.to_string(), t.tag.to_string()))
        .collect();
    tagged = tone_modifier.pre_merge_for_modify(tagged);

    let mut phones = Vec::new();
    let mut tones = Vec::new();
    let mut word2ph = Vec::new();

    for (word, pos) in tagged {
        if word.chars().all(|c| c.is_whitespace()) {
            let count = word.chars().count();
            word2ph.extend(std::iter::repeat(0).take(count));
            continue;
        }
        if word.trim().is_empty() {
            continue;
        }
        if english::is_english_token(&word) {
            let eng = english::g2p_word(&word);
            for (ph, tone) in eng.phones.iter().zip(eng.tones.iter()) {
                phones.push(ph.clone());
                tones.push(*tone);
            }
            word2ph.extend(eng.char_phone_counts);
            continue;
        }
        let mut syllables = get_syllables(&word);
        if syllables.is_empty() {
            continue;
        }
        if syllables
            .iter()
            .any(|s| s.final_with_tone.chars().any(|c| c.is_ascii_alphabetic()))
        {
            let finals: Vec<String> = syllables
                .iter()
                .map(|s| s.final_with_tone.clone())
                .collect();
            let adjusted = tone_modifier.modified_tone(&word, &pos, finals);
            for (syllable, final_with_tone) in syllables.iter_mut().zip(adjusted.into_iter()) {
                syllable.final_with_tone = final_with_tone;
            }
        }

        for syllable in syllables {
            let (char_phones, tone) = map_syllable_to_phones(&syllable).with_context(|| {
                format!(
                    "failed to map syllable '{}' in word '{}'",
                    syllable.ch, word
                )
            })?;
            let count = char_phones.len();
            phones.extend(char_phones);
            tones.extend(std::iter::repeat(tone).take(count));
            word2ph.push(count);
        }
    }

    Ok((phones, tones, word2ph))
}

struct SyllableInfo {
    ch: char,
    initial: String,
    final_with_tone: String,
}

fn get_syllables(word: &str) -> Vec<SyllableInfo> {
    word.chars()
        .zip(word.to_pinyin())
        .map(|(ch, opt)| match opt {
            Some(py) => {
                let (initial, final_with_tone) = split_initial_and_final(py, ch);
                SyllableInfo {
                    ch,
                    initial,
                    final_with_tone,
                }
            }
            None => {
                let fallback = ch.to_string();
                SyllableInfo {
                    ch,
                    initial: fallback.clone(),
                    final_with_tone: fallback,
                }
            }
        })
        .collect()
}

fn map_syllable_to_phones(info: &SyllableInfo) -> Result<(Vec<String>, i32)> {
    if info.initial == info.final_with_tone {
        return Ok((vec![info.ch.to_string()], 0));
    }

    let tone_char = info.final_with_tone.chars().last().unwrap_or('0');
    let tone = tone_char.to_digit(10).unwrap_or(0) as i32;

    let mut final_body = info
        .final_with_tone
        .trim_end_matches(|c: char| c.is_ascii_digit())
        .replace('ü', "v");

    if final_body == "lü" {
        final_body = "lv".to_string();
    }

    let mut pinyin = if info.initial.is_empty() {
        adjust_vowel_pinyin(final_body.clone())
    } else {
        adjust_consonant_pinyin(&info.initial, final_body.clone())
    };

    if pinyin.is_empty() {
        pinyin = info.ch.to_string();
    }

    if let Some(symbols) = PINYIN_TO_SYMBOL_MAP.get(&pinyin) {
        Ok((symbols.clone(), tone))
    } else if info.initial.is_empty() && is_punctuation_char(info.ch) {
        Ok((vec![info.ch.to_string()], 0))
    } else {
        bail!("no phone mapping for pinyin '{}'", pinyin)
    }
}

fn adjust_consonant_pinyin(initial: &str, final_body: String) -> String {
    let replaced = match final_body.as_str() {
        "uei" => "ui",
        "iou" => "iu",
        "uen" => "un",
        _ => final_body.as_str(),
    };
    format!("{}{}", initial, replaced)
}

fn adjust_vowel_pinyin(pinyin: String) -> String {
    match pinyin.as_str() {
        "ing" => "ying".to_string(),
        "i" => "yi".to_string(),
        "in" => "yin".to_string(),
        "u" => "wu".to_string(),
        _ => {
            if let Some(first) = pinyin.chars().next() {
                let rest: String = pinyin.chars().skip(1).collect();
                match first {
                    'v' => format!("yu{}", rest),
                    'e' => format!("e{}", rest),
                    'i' => format!("y{}", rest),
                    'u' => format!("w{}", rest),
                    _ => pinyin,
                }
            } else {
                pinyin
            }
        }
    }
}

fn is_punctuation_char(ch: char) -> bool {
    PUNCTUATIONS.iter().any(|p| p.chars().next() == Some(ch))
}

pub(crate) fn finals_with_tone(word: &str) -> Vec<String> {
    word.chars()
        .zip(word.to_pinyin())
        .map(|(ch, opt)| match opt {
            Some(py) => {
                let (_, finals) = split_initial_and_final(py, ch);
                finals
            }
            None => ch.to_string(),
        })
        .collect()
}

fn split_initial_and_final(py: Pinyin, fallback_char: char) -> (String, String) {
    let plain = py.plain();
    let with_tone_end = py.with_tone_num_end();
    let initial = extract_initial(plain);
    let finals = with_tone_end.get(initial.len()..).unwrap_or("").to_string();
    let finals = if finals.is_empty() {
        fallback_char.to_string()
    } else {
        finals
    };
    (initial.to_string(), finals)
}

fn extract_initial(plain: &str) -> &str {
    for candidate in PINYIN_INITIALS {
        if plain.starts_with(candidate) {
            return candidate;
        }
    }
    ""
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g2p_single_character() {
        let (phones, tones, word2ph) = g2p("你").expect("g2p succeeds");
        assert_eq!(phones, vec!["_", "n", "i", "_"]);
        assert_eq!(tones, vec![0, 3, 3, 0]);
        assert_eq!(word2ph, vec![1, 2, 1]);
    }

    #[test]
    fn g2p_applies_tone_sandhi() {
        let (phones, tones, word2ph) = g2p("你好").expect("g2p succeeds");
        assert_eq!(phones, vec!["_", "n", "i", "h", "ao", "_"]);
        assert_eq!(tones, vec![0, 2, 2, 3, 3, 0]);
        assert_eq!(word2ph, vec![1, 2, 2, 1]);
    }

    #[test]
    fn finals_with_tone_extraction() {
        assert_eq!(
            finals_with_tone("不对"),
            vec!["u4".to_string(), "ui4".to_string()]
        );
    }

    #[test]
    fn g2p_mixed_language() {
        let (phones, tones, word2ph) = g2p("Hello世界").expect("g2p succeeds");
        assert!(phones.iter().any(|p| p == "hh"));
        assert!(phones.iter().any(|p| p == "sh"));
        assert_eq!(phones.len(), tones.len());
        assert_eq!(word2ph.iter().sum::<usize>(), phones.len());
    }

    #[test]
    fn g2p_mixed_language_complex() {
        let text = "你好，欢迎使用风格语音合成Style-Bert-VITS2 ONNX TTS";
        let normalized = crate::nlp::chinese::normalizer::normalize_text(text);
        let (phones, _tones, word2ph) = g2p(&normalized).expect("g2p succeeds");
        let sum: usize = word2ph.iter().sum();
        assert_eq!(word2ph.len(), normalized.chars().count() + 2);
        assert_eq!(phones.len(), sum);
        assert!(phones.len() > 0);
    }

    #[test]
    fn g2p_english_sentence() {
        let text = "Occasionally give me gifts, and have special interactions with me on special holidays.";
        let normalized = crate::nlp::chinese::normalizer::normalize_text(text);
        let (phones, _tones, word2ph) = g2p(&normalized).expect("g2p succeeds");
        let sum: usize = word2ph.iter().sum();
        assert_eq!(word2ph.len(), normalized.chars().count() + 2);
        assert_eq!(phones.len(), sum);
        assert!(phones.len() > 0);
    }
}
