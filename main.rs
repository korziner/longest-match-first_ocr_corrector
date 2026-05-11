// OCR/HTR Corrector v2.6 — поддержка цифр и внутренней пунктуации в словах
// claude-opus-4-7-search_longest-match-first_ocr_corrector_v2_6
use serde::{Deserialize, Serialize};
use rayon::prelude::*;
use dashmap::DashMap;
use ahash::{AHashSet, RandomState};
use fastbloom::BloomFilter;
use std::io::{self, BufRead, Write, BufReader, BufWriter};
use std::fs::{self, File};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use colored::*;
use std::collections::HashMap;
use std::path::Path;
use rand::seq::SliceRandom;

#[derive(Parser, Debug)]
#[clap(name = "ocr-corrector", version = "2.6.0")]
struct Cli {
    #[clap(short, long)] input: Option<String>,
    #[clap(short, long, required = true)] dict: String,
    #[clap(short, long, default_value = "corrected.jsonl")] output: String,
    #[clap(short, long)] verbose: bool,
    #[clap(long)] analyze_only: bool,
    #[clap(long)] resume: bool,
    #[clap(long, default_value = "checkpoint.json.zst")] checkpoint: String,
    #[clap(long, default_value = "3000")] checkpoint_every: usize,
    #[clap(long)] fix_proper_nouns: bool,
    #[clap(long)] only_proper_nouns: bool,
    #[clap(long)] names_dict: Option<String>,
    #[clap(long, default_value = "5")] min_word_len: usize,
    #[clap(long, default_value = "2")] max_edit_dist: usize,
    #[clap(long, default_value = "3")] min_rule_freq: usize,
    #[clap(long, default_value = "8")] min_confusion_freq: usize,
    #[clap(long, default_value = "3")] min_ngram_freq: usize,
    #[clap(long, default_value = "7")] max_ngram: usize,
    #[clap(long, default_value = "1000")] batch_size: usize,
    #[clap(long, default_value = "25000000")] ngram_prune_limit: usize,
    #[clap(long, default_value = "30")] min_learned_pair_freq: usize,
    #[clap(long)] no_gpu: bool,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Record { text: String, #[serde(default)] title: String, #[serde(default)] date: String }
#[derive(Debug, Clone, Serialize)]
struct AlignDiff { wrong_text: String, right_text: String, pos_wrong: usize, pos_right: usize }
#[derive(Debug, Clone, Serialize)]
struct Correction { original: String, corrected: String, diffs: Vec<AlignDiff>, context: String, score: f64, reason: String, is_merge: bool }
#[derive(Debug, Clone)]
struct WordContext { clean: String, original_case: String, prefix_punct: String, suffix_punct: String, is_sentence_start: bool, is_after_abbrev: bool, next_word_is_initials: bool, prev_word_is_initials: bool }
#[derive(Debug, Clone, Serialize)]
struct Example { original: String, corrected: String, rule: String, score: f64, reason: String, context: String }

#[derive(Serialize, Deserialize)]
struct Checkpoint {
    processed: usize, total_oov: usize, total_matched: usize, skipped_proper_nouns: usize,
    error_matrix: Vec<(String, String, usize)>,
    ocr_confusions: Vec<(char, char, usize)>,
    ocr_insertions: Vec<(char, usize)>,
    ocr_deletions: Vec<(char, usize)>,
    ngrams: Vec<(Vec<String>, usize)>,
}

struct GlobalStats {
    error_matrix: DashMap<(String, String), usize, RandomState>,
    ocr_confusions: DashMap<(char, char), usize, RandomState>,
    ocr_insertions: DashMap<char, usize, RandomState>,
    ocr_deletions: DashMap<char, usize, RandomState>,
    ngrams: DashMap<Vec<String>, usize, RandomState>,
    examples_sub: DashMap<(char, char), Vec<Example>, RandomState>,
    examples_ins: DashMap<char, Vec<Example>, RandomState>,
    examples_del: DashMap<char, Vec<Example>, RandomState>,
    total_oov: AtomicUsize, total_matched: AtomicUsize, skipped_proper_nouns: AtomicUsize,
    fast_path_hits: AtomicUsize, slow_path_hits: AtomicUsize, processed: AtomicUsize,
}

impl GlobalStats {
    fn new() -> Self {
        Self {
            error_matrix: DashMap::with_hasher(RandomState::new()),
            ocr_confusions: DashMap::with_hasher(RandomState::new()),
            ocr_insertions: DashMap::with_hasher(RandomState::new()),
            ocr_deletions: DashMap::with_hasher(RandomState::new()),
            ngrams: DashMap::with_hasher(RandomState::new()),
            examples_sub: DashMap::with_hasher(RandomState::new()),
            examples_ins: DashMap::with_hasher(RandomState::new()),
            examples_del: DashMap::with_hasher(RandomState::new()),
            total_oov: AtomicUsize::new(0), total_matched: AtomicUsize::new(0),
            skipped_proper_nouns: AtomicUsize::new(0), fast_path_hits: AtomicUsize::new(0),
            slow_path_hits: AtomicUsize::new(0), processed: AtomicUsize::new(0),
        }
    }
    fn ngram_freq(&self, key: &[String]) -> usize { self.ngrams.get(key).map(|v| *v).unwrap_or(0) }
    fn prune_memory(&self, limit: usize) {
        if self.ngrams.len() > limit {
            let before = self.ngrams.len();
            self.ngrams.retain(|_, v| *v > 1);
            eprintln!("  🧹 Pruning: {} → {}", before, self.ngrams.len());
        }
    }
    fn save_checkpoint(&self, path: &str) -> io::Result<()> {
        let cp = Checkpoint {
            processed: self.processed.load(Ordering::SeqCst),
            total_oov: self.total_oov.load(Ordering::SeqCst),
            total_matched: self.total_matched.load(Ordering::SeqCst),
            skipped_proper_nouns: self.skipped_proper_nouns.load(Ordering::SeqCst),
            error_matrix: self.error_matrix.iter().map(|e| (e.key().0.clone(), e.key().1.clone(), *e.value())).collect(),
            ocr_confusions: self.ocr_confusions.iter().map(|e| (e.key().0, e.key().1, *e.value())).collect(),
            ocr_insertions: self.ocr_insertions.iter().map(|e| (*e.key(), *e.value())).collect(),
            ocr_deletions: self.ocr_deletions.iter().map(|e| (*e.key(), *e.value())).collect(),
            ngrams: self.ngrams.iter().map(|e| (e.key().clone(), *e.value())).collect(),
        };
        let tmp = format!("{}.tmp", path);
        let f = File::create(&tmp)?;
        let mut enc = zstd::Encoder::new(f, 3)?;
        serde_json::to_writer(&mut enc, &cp)?;
        enc.finish()?;
        fs::rename(&tmp, path)?;
        Ok(())
    }
    fn load_checkpoint(&self, path: &str) -> io::Result<()> {
        let f = File::open(path)?;
        let dec = zstd::Decoder::new(f)?;
        let cp: Checkpoint = serde_json::from_reader(dec)?;
        self.processed.store(cp.processed, Ordering::SeqCst);
        self.total_oov.store(cp.total_oov, Ordering::SeqCst);
        self.total_matched.store(cp.total_matched, Ordering::SeqCst);
        self.skipped_proper_nouns.store(cp.skipped_proper_nouns, Ordering::SeqCst);
        for (w,r,c) in cp.error_matrix { self.error_matrix.insert((w,r),c); }
        for (a,b,c) in cp.ocr_confusions { self.ocr_confusions.insert((a,b),c); }
        for (c,n) in cp.ocr_insertions { self.ocr_insertions.insert(c,n); }
        for (c,n) in cp.ocr_deletions { self.ocr_deletions.insert(c,n); }
        for (k,v) in cp.ngrams { self.ngrams.insert(k,v); }
        Ok(())
    }
}

struct Dictionary {
    bloom: BloomFilter,
    words: AHashSet<String>,
    by_prefix: HashMap<String, Vec<Arc<str>>>,
}

impl Dictionary {
    fn load(path: &str) -> io::Result<Self> {
        let meta = fs::metadata(path)?;
        let pb = ProgressBar::new(meta.len());
        pb.set_style(ProgressStyle::default_bar().template("  {elapsed_precise} 📚 {bar:30.cyan} {bytes}/{total_bytes} {msg}").unwrap());
        let file = File::open(path)?;
        let reader = BufReader::with_capacity(4*1024*1024, file);
        let mut all_words = Vec::with_capacity(1_600_000);
        let mut by_prefix: HashMap<String, Vec<Arc<str>>> = HashMap::new();
        for line in reader.lines() {
            let line = line?;
            pb.inc(line.len() as u64 + 1);
            let w = line.trim().to_lowercase();
            if w.chars().count() < 2 { continue; }
            if w.chars().count() >= 5 {
                let prefix: String = w.chars().take(2).collect();
                by_prefix.entry(prefix).or_default().push(Arc::from(w.as_str()));
            }
            all_words.push(w);
        }
        let mut bloom = BloomFilter::with_false_pos(0.001).expected_items(all_words.len());
        let mut words = AHashSet::with_capacity(all_words.len());
        for w in &all_words { bloom.insert(w.as_str()); words.insert(w.clone()); }
        pb.finish_with_message(format!("✓ {} слов", words.len()));
        Ok(Self { bloom, words, by_prefix })
    }
    fn contains(&self, word: &str) -> bool {
        let l = word.to_lowercase();
        self.bloom.contains(&l) && self.words.contains(&l)
    }
    fn candidates_for(&self, word_lower: &str) -> Option<&Vec<Arc<str>>> {
        // Берем первые 2 буквы (игнорируя цифры/пунктуацию) для поиска кандидатов
        let prefix: String = word_lower.chars().filter(|c| c.is_alphabetic()).take(2).collect();
        if prefix.chars().count() < 2 { return None; }
        self.by_prefix.get(&prefix)
    }
}

struct LearnedPairs {
    subs: AHashSet<(char, char)>,
    insertions: AHashSet<char>,
    deletions: AHashSet<char>,
}
impl LearnedPairs {
    fn build(stats: &GlobalStats, min_freq: usize) -> Self {
        Self {
            subs: stats.ocr_confusions.iter().filter(|e| *e.value() >= min_freq).map(|e| *e.key()).collect(),
            insertions: stats.ocr_insertions.iter().filter(|e| *e.value() >= min_freq).map(|e| *e.key()).collect(),
            deletions: stats.ocr_deletions.iter().filter(|e| *e.value() >= min_freq).map(|e| *e.key()).collect(),
        }
    }
    fn print_summary(&self, min_freq: usize) {
        eprintln!("\n  📐 Изученные правила (freq ≥ {}): subs={}, ins={}, del={}", min_freq, self.subs.len(), self.insertions.len(), self.deletions.len());
    }
}

// ═══ КЛАССИФИКАЦИЯ СИМВОЛОВ ═══
// "Букво-подобный" символ — буква, цифра ИЛИ внутренняя OCR-пунктуация
// Цифры (1,0,3,6,5,9) и пунктуация (;,:,!,|) могут быть OCR-ошибками внутри слов
fn is_word_inner(c: char) -> bool {
    c.is_alphabetic() || c.is_numeric() || matches!(c, ';' | ':' | '|' | '!' | '\\' | '/' | '*')
}
// "Внешняя" пунктуация — точно конец слова
fn is_outer_punct(c: char) -> bool {
    matches!(c, '.' | ',' | '?' | '"' | '\'' | '«' | '»' | '(' | ')' | '[' | ']' | '{' | '}' | '—' | '–')
}

fn is_acceptable_ocr_correction(wrong: &str, right: &str, learned: &LearnedPairs) -> bool {
    let wc: Vec<char> = wrong.chars().collect();
    let rc: Vec<char> = right.chars().collect();
    if (wc.len() as isize - rc.len() as isize).unsigned_abs() > 2 { return false; }
    let ops = build_alignment(&wc, &rc);
    let (mut ai, mut bi) = (0, 0);
    let mut total = 0;
    for op in ops {
        match op {
            AlignOp::Match => { ai += 1; bi += 1; }
            AlignOp::Sub => {
                if !learned.subs.contains(&(wc[ai], rc[bi])) { return false; }
                ai += 1; bi += 1; total += 1;
            }
            AlignOp::Del => {
                if !learned.deletions.contains(&wc[ai]) { return false; }
                ai += 1; total += 1;
            }
            AlignOp::Ins => {
                if !learned.insertions.contains(&rc[bi]) { return false; }
                bi += 1; total += 1;
            }
        }
    }
    if wc.len() > 4 && total > 2 { return false; }
    true
}

fn only_ending_changed(wrong: &str, right: &str) -> bool {
    let wc: Vec<char> = wrong.chars().collect();
    let rc: Vec<char> = right.chars().collect();
    let minl = wc.len().min(rc.len());
    if minl <= 3 { return false; }
    match wc.iter().zip(rc.iter()).position(|(a,b)| a != b) {
        None => wc.len() != rc.len(),
        Some(pos) => pos >= minl.saturating_sub(3),
    }
}

const SURNAME_SUFFIXES: &[&str] = &["овъ","евъ","инъ","ынъ","скій","цкій","ской","цкой","ова","ева","ина","ына","ская","цкая","ов","ев","ин","ын","ский","цкий"];
fn looks_like_surname(lower: &str) -> bool { SURNAME_SUFFIXES.iter().any(|s| lower.ends_with(s)) }
fn is_initials(word: &str) -> bool {
    let clean: String = word.chars().filter(|c| c.is_alphabetic()).collect();
    word.contains('.') && clean.chars().count() == 1 && clean.chars().next().map_or(false, |c| c.is_uppercase())
}
fn is_abbreviation(word: &str) -> bool {
    let lower = word.to_lowercase();
    ["г.","ул.","с.","д.","пр.","пер.","обл.","губ.","у."].iter().any(|a| lower.ends_with(a) || lower.as_str() == *a)
}
fn is_surname_in_context(words: &[String], i: usize) -> bool {
    if i+1 < words.len() && is_initials(&words[i+1]) { return true; }
    if i >= 1 && is_initials(&words[i-1]) { return true; }
    if i >= 2 && is_initials(&words[i-1]) && is_initials(&words[i-2]) { return true; }
    false
}
fn is_proper_noun(ctx: &WordContext, words: &[String], i: usize) -> bool {
    if ctx.original_case.is_empty() || ctx.original_case.chars().count() < 2 { return false; }
    let first = ctx.original_case.chars().next().unwrap();
    if !first.is_uppercase() { return false; }
    if ctx.is_sentence_start { return ctx.next_word_is_initials; }
    if ctx.next_word_is_initials || ctx.prev_word_is_initials { return true; }
    if ctx.is_after_abbrev { return true; }
    if is_surname_in_context(words, i) || looks_like_surname(&ctx.clean) { return true; }
    true
}
fn word_is_correctable(ctx: &WordContext, words: &[String], i: usize, dict: &Dictionary, cli: &Cli) -> bool {
    if dict.contains(&ctx.clean) { return false; }
    let proper = is_proper_noun(ctx, words, i);
    if proper {
        if cli.only_proper_nouns { return true; }
        if !cli.fix_proper_nouns { return false; }
    } else if cli.only_proper_nouns { return false; }
    true
}
fn correction_is_morphologically_safe(wrong: &str, right: &str, is_surname_ctx: bool) -> bool {
    if is_surname_ctx && only_ending_changed(wrong, right) {
        if looks_like_surname(wrong) && looks_like_surname(right) { return false; }
    }
    true
}
fn correction_is_acceptable(wrong: &str, right: &str, is_surname_ctx: bool, learned: &LearnedPairs) -> bool {
    is_acceptable_ocr_correction(wrong, right, learned) && correction_is_morphologically_safe(wrong, right, is_surname_ctx)
}

fn levenshtein(a: &[char], b: &[char]) -> usize {
    let (n, m) = (a.len(), b.len());
    if n == 0 { return m; }
    if m == 0 { return n; }
    let mut prev = (0..=m).collect::<Vec<_>>();
    let mut curr = vec![0; m+1];
    for i in 1..=n {
        curr[0] = i;
        for j in 1..=m {
            let cost = if a[i-1] == b[j-1] { 0 } else { 1 };
            curr[j] = (prev[j]+1).min(curr[j-1]+1).min(prev[j-1]+cost);
        }
        prev.copy_from_slice(&curr);
    }
    prev[m]
}

#[derive(Clone, Copy, PartialEq)]
enum AlignOp { Match, Sub, Ins, Del }

fn build_alignment(a: &[char], b: &[char]) -> Vec<AlignOp> {
    let (n, m) = (a.len(), b.len());
    let mut dp = vec![vec![0usize; m+1]; n+1];
    for i in 0..=n { dp[i][0] = i; }
    for j in 0..=m { dp[0][j] = j; }
    for i in 1..=n {
        for j in 1..=m {
            let cost = if a[i-1]==b[j-1] { 0 } else { 1 };
            dp[i][j] = (dp[i-1][j]+1).min(dp[i][j-1]+1).min(dp[i-1][j-1]+cost);
        }
    }
    let mut ops = Vec::new();
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        if i>0 && j>0 && a[i-1]==b[j-1] && dp[i][j]==dp[i-1][j-1] { ops.push(AlignOp::Match); i-=1; j-=1; }
        else if i>0 && j>0 && dp[i][j]==dp[i-1][j-1]+1 { ops.push(AlignOp::Sub); i-=1; j-=1; }
        else if i>0 && dp[i][j]==dp[i-1][j]+1 { ops.push(AlignOp::Del); i-=1; }
        else if j>0 && dp[i][j]==dp[i][j-1]+1 { ops.push(AlignOp::Ins); j-=1; }
        else {
            if i>0 && j>0 { ops.push(AlignOp::Sub); i-=1; j-=1; }
            else if i>0 { ops.push(AlignOp::Del); i-=1; }
            else { ops.push(AlignOp::Ins); j-=1; }
        }
    }
    ops.reverse();
    ops
}

fn extract_diffs(a: &[char], b: &[char]) -> Vec<AlignDiff> {
    let ops = build_alignment(a, b);
    let mut diffs = Vec::new();
    let (mut ai, mut bi) = (0, 0);
    let mut idx = 0;
    while idx < ops.len() {
        if ops[idx] == AlignOp::Match { ai+=1; bi+=1; idx+=1; continue; }
        let (pos_a, pos_b) = (ai, bi);
        let (mut wbuf, mut rbuf) = (String::new(), String::new());
        while idx < ops.len() && ops[idx] != AlignOp::Match {
            match ops[idx] {
                AlignOp::Sub => { wbuf.push(a[ai]); rbuf.push(b[bi]); ai+=1; bi+=1; }
                AlignOp::Del => { wbuf.push(a[ai]); ai+=1; }
                AlignOp::Ins => { rbuf.push(b[bi]); bi+=1; }
                AlignOp::Match => unreachable!(),
            }
            idx += 1;
        }
        diffs.push(AlignDiff { wrong_text: wbuf, right_text: rbuf, pos_wrong: pos_a, pos_right: pos_b });
    }
    diffs
}

fn collect_char_confusions(wrong: &[char], right: &[char], stats: &GlobalStats) {
    if wrong.len() == right.len() {
        for (wc, rc) in wrong.iter().zip(right.iter()) {
            if wc != rc { *stats.ocr_confusions.entry((*wc, *rc)).or_insert(0) += 1; }
        }
        return;
    }
    let ops = build_alignment(wrong, right);
    let (mut ai, mut bi) = (0, 0);
    for op in ops {
        match op {
            AlignOp::Match => { ai+=1; bi+=1; }
            AlignOp::Sub => { *stats.ocr_confusions.entry((wrong[ai], right[bi])).or_insert(0) += 1; ai+=1; bi+=1; }
            AlignOp::Del => { *stats.ocr_deletions.entry(wrong[ai]).or_insert(0) += 1; ai+=1; }
            AlignOp::Ins => { *stats.ocr_insertions.entry(right[bi]).or_insert(0) += 1; bi+=1; }
        }
    }
}

fn normalize_for_ngram(word: &str) -> String {
    let w = word.to_lowercase();
    let chars: Vec<char> = w.chars().filter(|c| c.is_alphabetic()).collect();
    let len = chars.len();
    if len <= 4 { return chars.iter().collect(); }
    let s: String = chars.iter().collect();
    let stem_len = if s.ends_with("ого")||s.ends_with("его")||s.ends_with("ему")||s.ends_with("ому") { len-3 }
    else if s.ends_with("ый")||s.ends_with("ій")||s.ends_with("ое")||s.ends_with("ая")||
            s.ends_with("ье")||s.ends_with("ья")||s.ends_with("ою")||s.ends_with("ею")||
            s.ends_with("емъ")||s.ends_with("омъ")||s.ends_with("амъ")||s.ends_with("ахъ")||
            s.ends_with("аго")||s.ends_with("яго") { len-2 }
    else if s.ends_with("ъ")||s.ends_with("ь")||s.ends_with("й")||
            s.ends_with("а")||s.ends_with("я")||s.ends_with("ы")||s.ends_with("и")||
            s.ends_with("у")||s.ends_with("ю")||s.ends_with("е") { len-1 }
    else { len };
    chars[..stem_len.min(len)].iter().collect()
}

fn collect_ngrams(stems: &[String], stats: &GlobalStats, max_n: usize) {
    for n in 2..=max_n {
        if stems.len() < n { break; }
        for window in stems.windows(n) {
            if window.iter().any(|s| s.len() < 2) { continue; }
            *stats.ngrams.entry(window.to_vec()).or_insert(0) += 1;
        }
    }
}

fn ngram_context_score(cand_stem: &str, prev_stems: &[String], next_stems: &[String], stats: &GlobalStats, min_freq: usize, max_n: usize) -> (f64, Option<String>) {
    let mut total = 0.0;
    let mut hit: Option<String> = None;
    let weight_for = |n: usize| -> f64 { match n { 7=>7.0, 6=>5.5, 5=>4.0, 4=>2.5, 3=>1.5, 2=>0.8, _=>(n as f64)*0.8 } };
    for n in (2..=max_n).rev() {
        let w = weight_for(n);
        let pl = n - 1;
        if prev_stems.len() >= pl {
            let mut key: Vec<String> = prev_stems[prev_stems.len()-pl..].to_vec();
            key.push(cand_stem.to_string());
            let freq = stats.ngram_freq(&key);
            if freq >= min_freq {
                total += (1.0+freq as f64).ln() * w;
                if hit.is_none() { hit = Some(format!("{}-gram «{}»:{}", n, key.join(" "), freq)); }
            }
        }
        let nl = n - 1;
        if next_stems.len() >= nl {
            let mut key = vec![cand_stem.to_string()];
            key.extend_from_slice(&next_stems[..nl]);
            let freq = stats.ngram_freq(&key);
            if freq >= min_freq {
                total += (1.0+freq as f64).ln() * w;
                if hit.is_none() { hit = Some(format!("{}-gram «{}»:{}", n, key.join(" "), freq)); }
            }
        }
    }
    (total, hit)
}

// ═══ ГЕНЕРАЦИЯ ВАРИАНТОВ С BOOTSTRAP DIGIT/PUNCT → LETTER ═══
fn generate_ocr_variants(word_lower: &str, stats: &GlobalStats, min_freq: usize) -> Vec<(String, char, char, usize)> {
    let chars: Vec<char> = word_lower.chars().collect();
    let mut variants: Vec<(String, char, char, usize)> = Vec::new();

    // Bootstrap: типичные OCR-ошибки графики (цифры и пунктуация → буквы)
    const BOOTSTRAP: &[(char, char)] = &[
        ('1','і'), ('1','i'), ('1','л'), ('1','т'),
        ('0','о'), ('0','c'),
        ('3','з'), ('3','з'),
        ('6','б'), ('5','б'),
        ('9','д'), ('8','в'),
        // Внутренняя пунктуация
        (';','щ'), (';','ц'),
        (':','і'), (':','и'),
        ('|','і'), ('|','л'),
        ('!','і'), ('!','л'),
        ('\\','л'), ('/','л'),
        ('*','ж'),
    ];
    for (i, &ch) in chars.iter().enumerate() {
        for &(from, to) in BOOTSTRAP {
            if ch == from {
                let mut nc = chars.clone();
                nc[i] = to;
                let v: String = nc.iter().collect::<String>();
                variants.push((v, from, to, i));
            }
        }
    }

    // Учёные конфузии из статистики
    let mut sorted: Vec<((char,char), usize)> = stats.ocr_confusions.iter()
        .filter(|e| *e.value() >= min_freq)
        .map(|e| (*e.key(), *e.value())).collect();
    sorted.sort_by(|a,b| b.1.cmp(&a.1));

    for ((from, to), _) in &sorted {
        for (i, &ch) in chars.iter().enumerate() {
            if ch == *from {
                let mut nc = chars.clone();
                nc[i] = *to;
                let v: String = nc.iter().collect::<String>();
                variants.push((v, *from, *to, i));
            }
        }
    }
    variants.sort_by(|a,b| a.0.cmp(&b.0));
    variants.dedup_by(|a,b| a.0 == b.0);
    variants
}

fn merge_split_words(words: &[String], dict: &Dictionary) -> (Vec<String>, Vec<Correction>) {
    let mut result = Vec::with_capacity(words.len());
    let mut corrections = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let mut found = false;
        for span in (2..=4).rev() {
            if i + span > words.len() { continue; }
            let chunks: Vec<String> = words[i..i+span].iter()
                .map(|w| w.chars().filter(|c| c.is_alphabetic()).collect::<String>()).collect();
            if chunks.iter().any(|c| c.is_empty()) { continue; }
            if !chunks.iter().all(|c| !dict.contains(&c.to_lowercase())) { continue; }
            let combined: String = chunks.concat();
            let combined_lower = combined.to_lowercase();
            if combined.chars().count() >= 5 && dict.contains(&combined_lower) {
                let original_text = words[i..i+span].join(" ");
                let prefix_punct: String = words[i].chars().take_while(|c| !is_word_inner(*c)).collect();
                let suffix_punct: String = words[i+span-1].chars().rev()
                    .take_while(|c| !is_word_inner(*c)).collect::<String>().chars().rev().collect();
                let first_upper = chunks[0].chars().next().map_or(false, |c| c.is_uppercase());
                let final_word = if first_upper {
                    let mut cs: Vec<char> = combined_lower.chars().collect();
                    if let Some(c) = cs.first_mut() { *c = c.to_uppercase().next().unwrap_or(*c); }
                    cs.into_iter().collect()
                } else { combined_lower };
                result.push(format!("{}{}{}", prefix_punct, final_word, suffix_punct));
                corrections.push(Correction {
                    original: original_text, corrected: final_word,
                    diffs: Vec::new(), context: String::new(),
                    score: 100.0, is_merge: true, reason: "склейка".to_string(),
                });
                i += span; found = true; break;
            }
        }
        if !found { result.push(words[i].clone()); i += 1; }
    }
    (result, corrections)
}

// ═══ КОНТЕКСТЫ — ИСПОЛЬЗУЕМ is_word_inner (включая цифры и внутр. пунктуацию) ═══
fn build_word_contexts(words: &[String]) -> Vec<WordContext> {
    let mut contexts = Vec::with_capacity(words.len());
    let mut prev_ends_sentence = true;
    for (i, word) in words.iter().enumerate() {
        // prefix_punct: внешняя пунктуация в начале (не цифры/буквы/внутр.пункт)
        let prefix_punct: String = word.chars().take_while(|c| !is_word_inner(*c)).collect();
        // suffix_punct: внешняя пунктуация в конце
        let suffix_punct: String = word.chars().rev().take_while(|c| !is_word_inner(*c))
            .collect::<String>().chars().rev().collect();
        // original_case: буквы + цифры + внутренняя пунктуация (для OCR-исправления)
        let original_case: String = word.chars().filter(|c| is_word_inner(*c)).collect();
        let clean = original_case.to_lowercase();

        let is_sentence_start = prev_ends_sentence;
        let is_after_abbrev = i > 0 && is_abbreviation(&words[i-1]);
        let next_word_is_initials = i+1 < words.len() && is_initials(&words[i+1]);
        let prev_word_is_initials = i > 0 && is_initials(&words[i-1]);

        contexts.push(WordContext {
            clean, original_case, prefix_punct,
            suffix_punct: suffix_punct.clone(),
            is_sentence_start, is_after_abbrev,
            next_word_is_initials, prev_word_is_initials,
        });
        prev_ends_sentence = suffix_punct.contains('.') || suffix_punct.contains('!') || suffix_punct.contains('?');
    }
    contexts
}

fn restore_case(original: &str, corrected_lower: &str) -> String {
    let orig_chars: Vec<char> = original.chars().collect();
    let mut corr_chars: Vec<char> = corrected_lower.chars().collect();
    for (i, cc) in corr_chars.iter_mut().enumerate() {
        if i < orig_chars.len() && orig_chars[i].is_uppercase() {
            *cc = cc.to_uppercase().next().unwrap_or(*cc);
        }
    }
    if orig_chars.first().map_or(false, |c| c.is_uppercase()) {
        if let Some(c) = corr_chars.first_mut() {
            *c = c.to_uppercase().next().unwrap_or(*c);
        }
    }
    corr_chars.into_iter().collect()
}

fn add_example(stats: &GlobalStats, from: char, to: char, original: String, corrected: String, reason: String, context: String, score: f64) {
    let ex = Example { original, corrected, rule: format!("{}→{}", from, to), score, reason, context };
    let mut examples = stats.examples_sub.entry((from, to)).or_default();
    examples.push(ex);
    if examples.len() > 50 {
        examples.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        examples.truncate(50);
    }
}

fn print_intermediate_ocr_stats(stats: &GlobalStats) {
    let mut subs: Vec<_> = stats.ocr_confusions.iter().map(|e| (*e.key(), *e.value())).collect();
    subs.sort_by(|a,b| b.1.cmp(&a.1));
    let mut ins: Vec<_> = stats.ocr_insertions.iter().map(|e| (*e.key(), *e.value())).collect();
    ins.sort_by(|a,b| b.1.cmp(&a.1));
    let mut dels: Vec<_> = stats.ocr_deletions.iter().map(|e| (*e.key(), *e.value())).collect();
    dels.sort_by(|a,b| b.1.cmp(&a.1));
    eprintln!("\n  📊 Топ OCR-правил:");
    let s_str = subs.iter().take(12).map(|((f,t),c)| format!("{}→{}({})", f, t, c)).collect::<Vec<_>>().join(", ");
    eprintln!("    Sub: {}", s_str);
    let i_str = ins.iter().take(8).map(|(c,n)| format!("+{}({})", c, n)).collect::<Vec<_>>().join(", ");
    eprintln!("    Ins: {}", i_str);
    let d_str = dels.iter().take(8).map(|(c,n)| format!("-{}({})", c, n)).collect::<Vec<_>>().join(", ");
    eprintln!("    Del: {}", d_str);
}

fn print_intermediate_examples(stats: &GlobalStats) {
    eprintln!("\n  📋 Случайные примеры в контексте:");
    let mut subs: Vec<_> = stats.ocr_confusions.iter().map(|e| (*e.key(), *e.value())).collect();
    subs.sort_by(|a,b| b.1.cmp(&a.1));
    let mut rng = rand::thread_rng();
    for ((from, to), freq) in subs.iter().take(8) {
        if let Some(examples) = stats.examples_sub.get(&(*from, *to)) {
            let slice: &[Example] = &*examples;
            if !slice.is_empty() {
                if let Some(ex) = slice.choose(&mut rng) {
                    eprintln!("  {:>6} «{}»→«{}» | {} | {}", freq,
                        from.to_string().red(), to.to_string().green(),
                        ex.original, ex.context.dimmed());
                }
            }
        }
    }
    eprintln!();
}

fn find_best_correction(
    clean: &str, original_case: &str, is_surname_ctx: bool,
    prev_stems: &[String], next_stems: &[String],
    dict: &Dictionary, stats: &GlobalStats, cli: &Cli, learned: &LearnedPairs,
) -> Option<(String, Vec<AlignDiff>, f64, String)> {
    let clean_chars: Vec<char> = clean.chars().collect();
    let mut best: Option<(String, Vec<AlignDiff>, f64, String)> = None;

    // FAST PATH: confusion-driven (включая bootstrap)
    let variants = generate_ocr_variants(clean, stats, cli.min_confusion_freq);
    for (variant, from_ch, to_ch, pos) in variants {
        if !dict.contains(&variant) { continue; }
        if !correction_is_acceptable(clean, &variant, is_surname_ctx, learned) {
            // Для bootstrap-пар (цифры/пунктуация) пропускаем фильтр learned, если from — цифра/пунктуация
            if !(!from_ch.is_alphabetic() && to_ch.is_alphabetic()) { continue; }
        }
        let conf_freq = stats.ocr_confusions.get(&(from_ch, to_ch)).map(|v| *v as f64).unwrap_or(1.0);
        let mut score = 50.0 + (1.0 + conf_freq).ln() * 4.0;
        // Бонус за исправление цифры/пунктуации в букву (это почти всегда правильно)
        if !from_ch.is_alphabetic() && to_ch.is_alphabetic() {
            score += 30.0;
        }
        let cand_stem = normalize_for_ngram(&variant);
        let (ctx_bonus, hit) = ngram_context_score(&cand_stem, prev_stems, next_stems, stats, cli.min_ngram_freq, cli.max_ngram);
        score += ctx_bonus;
        let var_chars: Vec<char> = variant.chars().collect();
        let diffs = extract_diffs(&clean_chars, &var_chars);
        let corrected = restore_case(original_case, &variant);
        let mut reason = format!("OCR «{}»→«{}» поз.{} freq={}", from_ch, to_ch, pos+1, conf_freq as usize);
        if let Some(h) = hit { reason = format!("{} + {}", reason, h); }
        if best.as_ref().map_or(true, |(_,_,s,_)| score > *s) {
            best = Some((corrected, diffs, score, reason));
        }
    }
    if best.is_some() {
        stats.fast_path_hits.fetch_add(1, Ordering::Relaxed);
        return best;
    }

    // SLOW PATH
    let candidates = dict.candidates_for(clean)?;
    let clen = clean_chars.len();
    for cand in candidates {
        let cand_chars: Vec<char> = cand.chars().collect();
        if (clen as isize - cand_chars.len() as isize).unsigned_abs() > cli.max_edit_dist { continue; }
        let dist = levenshtein(&clean_chars, &cand_chars);
        if dist == 0 || dist > cli.max_edit_dist { continue; }
        if !correction_is_acceptable(clean, cand, is_surname_ctx, learned) { continue; }
        let diffs = extract_diffs(&clean_chars, &cand_chars);
        let mut score = 0.0;
        let mut has_support = false;
        let mut unsupported = 0;
        for diff in &diffs {
            let key = (diff.wrong_text.clone(), diff.right_text.clone());
            if let Some(freq) = stats.error_matrix.get(&key) {
                let f = *freq;
                if f >= cli.min_rule_freq {
                    let nlen = diff.wrong_text.chars().count().max(diff.right_text.chars().count()) as f64;
                    score += (1.0 + f as f64).ln() * nlen;
                    has_support = true;
                } else { unsupported += 1; }
            } else { unsupported += 1; }
        }
        score -= unsupported as f64 * 3.0;
        score -= dist as f64 * 0.5;
        let cand_stem = normalize_for_ngram(cand);
        let (ctx_bonus, hit) = ngram_context_score(&cand_stem, prev_stems, next_stems, stats, cli.min_ngram_freq, cli.max_ngram);
        score += ctx_bonus;
        if (has_support || ctx_bonus > 0.0) && score > 0.0 {
            if best.as_ref().map_or(true, |(_,_,s,_)| score > *s) {
                let corrected = restore_case(original_case, cand);
                let mut reason = format!("n-gram dist={}", dist);
                if let Some(h) = hit { reason = format!("{} + {}", reason, h); }
                best = Some((corrected, diffs, score, reason));
            }
        }
    }
    if best.is_some() { stats.slow_path_hits.fetch_add(1, Ordering::Relaxed); }
    best
}

fn pass1_process_text(text: &str, dict: &Dictionary, stats: &GlobalStats, cli: &Cli) {
    let raw_words: Vec<String> = text.split_whitespace().map(String::from).collect();
    let (merged, _) = merge_split_words(&raw_words, dict);

    // N-граммы — только из чистых букв
    let mut buf: Vec<String> = Vec::new();
    for w in &merged {
        let clean: String = w.chars().filter(|c| c.is_alphabetic()).collect::<String>().to_lowercase();
        if !clean.is_empty() && dict.contains(&clean) {
            buf.push(normalize_for_ngram(&clean));
        } else {
            if buf.len() >= 2 { collect_ngrams(&buf, stats, cli.max_ngram); }
            buf.clear();
        }
    }
    if buf.len() >= 2 { collect_ngrams(&buf, stats, cli.max_ngram); }

    let contexts = build_word_contexts(&merged);
    for (i, ctx) in contexts.iter().enumerate() {
        if dict.contains(&ctx.clean) { continue; }
        if ctx.clean.chars().count() < cli.min_word_len { continue; }
        let proper = is_proper_noun(ctx, &merged, i);
        if cli.only_proper_nouns && !proper { continue; }
        if !cli.fix_proper_nouns && !cli.only_proper_nouns && proper {
            stats.skipped_proper_nouns.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        stats.total_oov.fetch_add(1, Ordering::Relaxed);

        let chars: Vec<char> = ctx.clean.chars().collect();
        let candidates = match dict.candidates_for(&ctx.clean) { Some(c)=>c, None=>continue };

        let mut best_dist = cli.max_edit_dist + 1;
        let mut best_word: Option<&str> = None;
        let surname_ctx = looks_like_surname(&ctx.clean) || is_surname_in_context(&merged, i);

        for cand in candidates {
            let cc: Vec<char> = cand.chars().collect();
            if (chars.len() as isize - cc.len() as isize).unsigned_abs() > cli.max_edit_dist { continue; }
            let dist = levenshtein(&chars, &cc);
            if dist > 0 && dist < best_dist {
                if !correction_is_morphologically_safe(&ctx.clean, cand, surname_ctx) { continue; }
                best_dist = dist;
                best_word = Some(cand);
                if dist == 1 { break; }
            }
        }

        if let Some(correct) = best_word {
            stats.total_matched.fetch_add(1, Ordering::Relaxed);
            let correct_chars: Vec<char> = correct.chars().collect();
            let diffs = extract_diffs(&chars, &correct_chars);
            for diff in &diffs {
                *stats.error_matrix.entry((diff.wrong_text.clone(), diff.right_text.clone())).or_insert(0) += 1;
            }
            if best_dist <= 2 {
                collect_char_confusions(&chars, &correct_chars, stats);
                if diffs.len() == 1 && diffs[0].wrong_text.chars().count() == 1 && diffs[0].right_text.chars().count() == 1 {
                    let from = diffs[0].wrong_text.chars().next().unwrap();
                    let to = diffs[0].right_text.chars().next().unwrap();
                    let context_str = raw_words[i.saturating_sub(3)..(i+4).min(raw_words.len())].join(" ");
                    add_example(stats, from, to, ctx.original_case.clone(),
                        restore_case(&ctx.original_case, correct), "Pass1".into(), context_str, 1.0);
                }
            }
        }
    }
}

fn run_pass1(input_path: &str, dict: &Dictionary, stats: &GlobalStats, cli: &Cli, total_lines: u64) -> io::Result<()> {
    let pb = ProgressBar::new(total_lines);
    pb.set_style(ProgressStyle::default_bar()
        .template("  {elapsed_precise} Pass 1 [{bar:30.cyan}] {percent:>3}% {pos}/{len} | {msg}").unwrap());
    let skip_count = stats.processed.load(Ordering::SeqCst);
    if skip_count > 0 {
        eprintln!("  ♻ Пропуск {} строк", skip_count);
        pb.set_position(skip_count as u64);
    }
    let reader = open_reader(input_path)?;
    let mut batch: Vec<Record> = Vec::with_capacity(cli.batch_size);
    let mut line_idx = 0;
    for line in reader.lines() {
        let line = line?;
        line_idx += 1;
        if line_idx <= skip_count { continue; }
        if let Some(rec) = parse_line(&line) { batch.push(rec); }
        if batch.len() >= cli.batch_size {
            let prev_proc = stats.processed.load(Ordering::SeqCst);
            batch.par_iter().for_each(|rec| pass1_process_text(&rec.text, dict, stats, cli));
            stats.processed.fetch_add(batch.len(), Ordering::SeqCst);
            batch.clear();
            stats.prune_memory(cli.ngram_prune_limit);
            let cur_proc = stats.processed.load(Ordering::SeqCst);
            pb.set_position(line_idx as u64);
            pb.set_message(format!("OOV:{} матр:{} конф:{} ngram:{}",
                stats.total_oov.load(Ordering::Relaxed),
                stats.error_matrix.len(), stats.ocr_confusions.len(), stats.ngrams.len()));
            if prev_proc / cli.checkpoint_every < cur_proc / cli.checkpoint_every {
                pb.suspend(|| {
                    print_intermediate_ocr_stats(stats);
                    print_intermediate_examples(stats);
                    if let Err(e) = stats.save_checkpoint(&cli.checkpoint) {
                        eprintln!("  ⚠ Чекпоинт: {}", e);
                    } else {
                        eprintln!("  💾 Чекпоинт сохранён ({} строк)\n", cur_proc);
                    }
                });
            }
        }
    }
    if !batch.is_empty() {
        batch.par_iter().for_each(|rec| pass1_process_text(&rec.text, dict, stats, cli));
        stats.processed.fetch_add(batch.len(), Ordering::SeqCst);
    }
    stats.save_checkpoint(&cli.checkpoint)?;
    pb.finish_with_message(format!("✓ {} зап.", stats.processed.load(Ordering::Relaxed)));
    Ok(())
}

fn pass2_correct_record(rec: &mut Record, dict: &Dictionary, stats: &GlobalStats, cli: &Cli, learned: &LearnedPairs) -> Vec<Correction> {
    let raw_words: Vec<String> = rec.text.split_whitespace().map(String::from).collect();
    let (mut merged, mut corrections) = merge_split_words(&raw_words, dict);
    let contexts = build_word_contexts(&merged);
    let ctx_snap = merged.clone();
    let len = merged.len();
    let stems: Vec<String> = ctx_snap.iter().map(|w| {
        let clean: String = w.chars().filter(|c| c.is_alphabetic()).collect::<String>().to_lowercase();
        normalize_for_ngram(&clean)
    }).collect();

    for (i, ctx) in contexts.iter().enumerate() {
        if ctx.clean.chars().count() < cli.min_word_len { continue; }
        if !word_is_correctable(ctx, &ctx_snap, i, dict, cli) { continue; }
        let is_surname_ctx = looks_like_surname(&ctx.clean) || is_surname_in_context(&ctx_snap, i);
        let prev_stems: Vec<String> = if i >= 6 { stems[i-6..i].to_vec() } else { stems[..i].to_vec() };
        let next_stems: Vec<String> = stems[(i+1)..(i+7).min(len)].to_vec();
        if let Some((corrected, diffs, score, reason)) = find_best_correction(
            &ctx.clean, &ctx.original_case, is_surname_ctx,
            &prev_stems, &next_stems, dict, stats, cli, learned,
        ) {
            let ctx_start = i.saturating_sub(3);
            let ctx_end = (i+4).min(len);
            let context = ctx_snap[ctx_start..ctx_end].join(" ");
            corrections.push(Correction {
                original: ctx.original_case.clone(),
                corrected: corrected.clone(),
                diffs, context, score, is_merge: false, reason,
            });
            merged[i] = format!("{}{}{}", ctx.prefix_punct, corrected, ctx.suffix_punct);
        }
    }
    rec.text = merged.join(" ");
    corrections
}

fn run_pass2(input_path: &str, dict: &Dictionary, stats: &GlobalStats, cli: &Cli, learned: &LearnedPairs, total_lines: u64) -> io::Result<(usize, usize)> {
    let pb = ProgressBar::new(total_lines);
    pb.set_style(ProgressStyle::default_bar()
        .template("  {elapsed_precise} Pass 2 [{bar:30.magenta}] {percent:>3}% {pos}/{len} | {msg}").unwrap());

    let pass2_progress = "pass2_last_line.txt";
    let mut start_from: usize = 0;
    if cli.resume && Path::new(pass2_progress).exists() {
        if let Ok(s) = std::fs::read_to_string(pass2_progress) {
            if let Ok(n) = s.trim().parse::<usize>() {
                start_from = n;
                eprintln!("  ♻ Pass 2 продолжаем со строки {}", n);
            }
        }
    }

    let reader = open_reader(input_path)?;
    let mut writer = BufWriter::with_capacity(8*1024*1024, File::create(&cli.output)?);
    let mut pairs_file = BufWriter::new(File::create("pairs.accepted.tsv")?);
    let mut merges_file = BufWriter::new(File::create("merges.log")?);
    writeln!(pairs_file, "original\tcorrected\tscore\trule\treason\tcontext")?;
    writeln!(merges_file, "original\tcorrected\tcontext")?;
    let mut total_records = 0;
    let mut total_corrections = 0;
    let mut raw_batch: Vec<String> = Vec::with_capacity(cli.batch_size);
    let mut line_idx = 0;

    let flush_batch = |raw_batch: &mut Vec<String>,
                       writer: &mut BufWriter<File>,
                       pairs_file: &mut BufWriter<File>,
                       merges_file: &mut BufWriter<File>,
                       total_records: &mut usize,
                       total_corrections: &mut usize| -> io::Result<()> {
        let results: Vec<(Record, Vec<Correction>)> = raw_batch.par_iter()
            .filter_map(|l| parse_line(l))
            .map(|mut rec| {
                let corrs = pass2_correct_record(&mut rec, dict, stats, cli, learned);
                (rec, corrs)
            }).collect();
        for (rec, corrections) in results {
            for c in &corrections {
                if c.is_merge {
                    writeln!(merges_file, "{}\t{}\t{}", c.original, c.corrected, c.context.replace('\t', " "))?;
                } else {
                    let rule: String = c.diffs.iter()
                        .map(|d| format!("«{}»→«{}»", d.wrong_text, d.right_text))
                        .collect::<Vec<_>>().join("; ");
                    writeln!(pairs_file, "{}\t{}\t{:.2}\t{}\t{}\t{}",
                        c.original, c.corrected, c.score, rule, c.reason,
                        c.context.replace('\t', " "))?;
                }
            }
            if cli.verbose && !corrections.is_empty() {
                print_corrections_verbose(&rec, &corrections);
            }
            *total_corrections += corrections.len();
            *total_records += 1;
            writeln!(writer, "{}", serde_json::to_string(&rec)?)?;
        }
        pb.inc(raw_batch.len() as u64);
        pb.set_message(format!("испр:{} fast:{} slow:{}",
            *total_corrections,
            stats.fast_path_hits.load(Ordering::Relaxed),
            stats.slow_path_hits.load(Ordering::Relaxed)));
        raw_batch.clear();
        Ok(())
    };

    for line in reader.lines() {
        let l = line?;
        line_idx += 1;
        if line_idx <= start_from { continue; }
        raw_batch.push(l);
        if raw_batch.len() >= cli.batch_size {
            flush_batch(&mut raw_batch, &mut writer, &mut pairs_file, &mut merges_file,
                &mut total_records, &mut total_corrections)?;
            let _ = std::fs::write(pass2_progress, line_idx.to_string());
        }
    }
    if !raw_batch.is_empty() {
        flush_batch(&mut raw_batch, &mut writer, &mut pairs_file, &mut merges_file,
            &mut total_records, &mut total_corrections)?;
    }
    if Path::new(pass2_progress).exists() { let _ = fs::remove_file(pass2_progress); }
    writer.flush()?; pairs_file.flush()?; merges_file.flush()?;
    pb.finish_with_message(format!("✓ {} зап. {} испр.", total_records, total_corrections));
    Ok((total_records, total_corrections))
}

fn print_corrections_verbose(rec: &Record, corrections: &[Correction]) {
    let title = if rec.title.is_empty() { "—" } else { &rec.title };
    println!("{}", format!("┌─ {} ", title).blue().bold());
    for c in corrections {
        if c.is_merge {
            println!("│ [СКЛЕЙКА] «{}» → «{}»", c.original.red(), c.corrected.green());
            println!("│"); continue;
        }
        let diff_summary: String = c.diffs.iter()
            .map(|d| format!("«{}»→«{}»", d.wrong_text, d.right_text))
            .collect::<Vec<_>>().join("; ");
        println!("│ {} {} {}",
            diff_summary.yellow(),
            format!("(score {:.1})", c.score).dimmed(),
            format!("[{}]", c.reason).cyan().dimmed());
        let orig_chars: Vec<char> = c.original.chars().collect();
        let corr_chars: Vec<char> = c.corrected.chars().collect();
        let mut orig_hl = vec![false; orig_chars.len()];
        let mut corr_hl = vec![false; corr_chars.len()];
        for d in &c.diffs {
            let ew = d.pos_wrong + d.wrong_text.chars().count();
            for p in d.pos_wrong..ew.min(orig_chars.len()) { orig_hl[p] = true; }
            let er = d.pos_right + d.right_text.chars().count();
            for p in d.pos_right..er.min(corr_chars.len()) { corr_hl[p] = true; }
        }
        print!("│  ");
        for (i, ch) in orig_chars.iter().enumerate() {
            if orig_hl[i] { print!("{}", ch.to_string().red().bold().underline()); }
            else { print!("{}", ch.to_string().dimmed()); }
        }
        println!();
        print!("│  ");
        for (i, ch) in corr_chars.iter().enumerate() {
            if corr_hl[i] { print!("{}", ch.to_string().green().bold().underline()); }
            else { print!("{}", ch.to_string().dimmed()); }
        }
        println!();
        print!("│  ");
        for i in 0..orig_chars.len().max(corr_chars.len()) {
            let a = i < orig_hl.len() && orig_hl[i];
            let b = i < corr_hl.len() && corr_hl[i];
            if a || b { print!("{}", "↕".yellow()); } else { print!(" "); }
        }
        println!();
        if !c.context.is_empty() { println!("│  {}", c.context.italic().dimmed()); }
        println!("│");
    }
    println!("{}", "└─".blue());
}

fn save_artifacts(stats: &GlobalStats, cli: &Cli) -> io::Result<()> {
    let mut entries: Vec<((String,String),usize)> = stats.error_matrix.iter()
        .map(|e| (e.key().clone(), *e.value())).collect();
    entries.sort_by(|a,b| b.1.cmp(&a.1));
    let map: HashMap<String,usize> = entries.iter()
        .map(|((w,r),c)| (format!("{} -> {}", w, r), *c)).collect();
    fs::write("matrix.pass1.raw.json", serde_json::to_string_pretty(&map)?)?;

    let mut confs: Vec<((char,char),usize)> = stats.ocr_confusions.iter()
        .map(|e| (*e.key(), *e.value())).collect();
    confs.sort_by(|a,b| b.1.cmp(&a.1));
    let mut csv = BufWriter::new(File::create("ocr_confusions.csv")?);
    writeln!(csv, "char_from\tchar_to\tfrequency")?;
    for ((f,t),fr) in &confs { writeln!(csv, "{}\t{}\t{}", f, t, fr)?; }
    csv.flush()?;

    let mut inss: Vec<(char,usize)> = stats.ocr_insertions.iter().map(|e| (*e.key(), *e.value())).collect();
    inss.sort_by(|a,b| b.1.cmp(&a.1));
    let mut ins_csv = BufWriter::new(File::create("ocr_insertions.csv")?);
    writeln!(ins_csv, "char_inserted\tfrequency")?;
    for (c, fr) in &inss { writeln!(ins_csv, "{}\t{}", c, fr)?; }
    ins_csv.flush()?;

    let mut dels: Vec<(char,usize)> = stats.ocr_deletions.iter().map(|e| (*e.key(), *e.value())).collect();
    dels.sort_by(|a,b| b.1.cmp(&a.1));
    let mut del_csv = BufWriter::new(File::create("ocr_deletions.csv")?);
    writeln!(del_csv, "char_deleted\tfrequency")?;
    for (c, fr) in &dels { writeln!(del_csv, "{}\t{}", c, fr)?; }
    del_csv.flush()?;

    let mut by_len: HashMap<usize, Vec<(Vec<String>,usize)>> = HashMap::new();
    for entry in stats.ngrams.iter() {
        if *entry.value() < cli.min_ngram_freq { continue; }
        by_len.entry(entry.key().len()).or_default().push((entry.key().clone(), *entry.value()));
    }
    let mut out = BufWriter::new(File::create("ngrams_top.tsv")?);
    writeln!(out, "n\tfrequency\tngram")?;
    for n in (2..=cli.max_ngram).rev() {
        if let Some(mut v) = by_len.remove(&n) {
            v.sort_by(|a,b| b.1.cmp(&a.1));
            for (gram, freq) in v.iter().take(500) {
                writeln!(out, "{}\t{}\t{}", n, freq, gram.join(" "))?;
            }
        }
    }
    out.flush()?;

    let mut ex_sub = BufWriter::new(File::create("examples.sub.tsv")?);
    writeln!(ex_sub, "original\tcorrected\trule\tscore\treason\tcontext")?;
    for ((from, to), _) in confs.iter().take(50) {
        if let Some(examples) = stats.examples_sub.get(&(*from, *to)) {
            for ex in examples.iter().take(5) {
                writeln!(ex_sub, "{}\t{}\t{}→{}\t{:.2}\t{}\t{}",
                    ex.original, ex.corrected, from, to, ex.score, ex.reason, ex.context)?;
            }
        }
    }
    ex_sub.flush()?;
    Ok(())
}

fn open_reader(path: &str) -> io::Result<Box<dyn BufRead>> {
    let file = File::open(path)?;
    if path.ends_with(".zst") {
        Ok(Box::new(BufReader::with_capacity(8*1024*1024, zstd::Decoder::new(file)?)))
    } else {
        Ok(Box::new(BufReader::with_capacity(8*1024*1024, file)))
    }
}

fn count_lines(path: &str) -> io::Result<u64> {
    let reader = open_reader(path)?;
    Ok(reader.lines().count() as u64)
}

fn buffer_stdin(tmp: &str) -> io::Result<()> {
    let mut f = BufWriter::with_capacity(8*1024*1024, File::create(tmp)?);
    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::default_spinner()
        .template("  {elapsed_precise} {spinner:.yellow} STDIN → {msg}").unwrap());
    pb.set_message(tmp.to_string());
    let stdin = io::stdin();
    let mut lines = 0u64;
    for line in stdin.lock().lines() {
        let l = line?; lines += 1;
        writeln!(f, "{}", l)?;
        if lines % 5000 == 0 { pb.set_message(format!("{}: {} строк", tmp, lines)); }
    }
    f.flush()?;
    pb.finish_with_message(format!("✓ {} строк", lines));
    Ok(())
}

fn parse_line(line: &str) -> Option<Record> {
    let t = line.trim();
    if t.is_empty() || t.starts_with("--") || t.starts_with('#') { return None; }
    if let Ok(rec) = serde_json::from_str::<Record>(t) {
        if !rec.text.is_empty() { return Some(rec); }
    }
    if let Some(text) = extract_field(t, "text") {
        if text.len() > 5 {
            return Some(Record {
                text,
                title: extract_field(t, "title").unwrap_or_default(),
                date: extract_field(t, "date").unwrap_or_default(),
            });
        }
    }
    if !t.starts_with('{') && t.chars().count() > 20 {
        return Some(Record { text: t.to_string(), title: String::new(), date: String::new() });
    }
    None
}

fn extract_field(json: &str, field: &str) -> Option<String> {
    for pat in [format!("\"{}\":\"", field), format!("\"{}\": \"", field)].iter() {
        if let Some(start) = json.find(pat.as_str()) {
            let vb = start + pat.len();
            let rem = &json[vb..];
            let end = ["\", \"", "\"}"].iter()
                .filter_map(|t| rem.find(t)).min()
                .unwrap_or(rem.len());
            return Some(rem[..end].replace("\\n", "\n").replace("\\\"", "\""));
        }
    }
    None
}

fn main() -> io::Result<()> {
    let cli = Cli::parse();
    eprintln!("{}", "═══ OCR/HTR Corrector v2.6 — цифры + внутр. пунктуация ═══".cyan().bold());
    eprintln!("  ✓ Слово в словаре → не трогать");
    eprintln!("  ✓ Цифры (1,0,3,6,5,9) и пунктуация (;:|!) → буквы");
    eprintln!("  ✓ Защита фамилий и морфологии");
    eprintln!("  ✓ Параллельный Pass 2 (rayon, batch={})", cli.batch_size);

    let dict = Arc::new(Dictionary::load(&cli.dict)?);
    let stats = Arc::new(GlobalStats::new());

    if cli.resume && Path::new(&cli.checkpoint).exists() {
        eprintln!("  ♻ Загрузка чекпоинта {}", cli.checkpoint);
        match stats.load_checkpoint(&cli.checkpoint) {
            Ok(_) => eprintln!("  ✓ Восстановлено: {} строк", stats.processed.load(Ordering::Relaxed)),
            Err(e) => eprintln!("  ⚠ Ошибка: {}", e),
        }
    }

    let input_path = match &cli.input {
        Some(p) => p.clone(),
        None => { let tmp = "_ocr_temp.jsonl"; buffer_stdin(tmp)?; tmp.to_string() }
    };

    eprintln!("\n📊 Подсчёт строк...");
    let total_lines = count_lines(&input_path)?;
    eprintln!("  ✓ {} строк", total_lines);

    eprintln!("\n─── Pass 1 ───");
    run_pass1(&input_path, &dict, &stats, &cli, total_lines)?;
    save_artifacts(&stats, &cli)?;

    let learned = LearnedPairs::build(&stats, cli.min_learned_pair_freq);
    learned.print_summary(cli.min_learned_pair_freq);

    if cli.analyze_only {
        eprintln!("\n✓ Анализ завершён");
        return Ok(());
    }

    eprintln!("\n─── Pass 2 ───");
    let (recs, corrs) = run_pass2(&input_path, &dict, &stats, &cli, &learned, total_lines)?;

    eprintln!("\n═══ Готово ═══");
    eprintln!("  📄 {} ({} записей, {} исправлений)", cli.output, recs, corrs);
    eprintln!("  ⚡ fast: {}", stats.fast_path_hits.load(Ordering::Relaxed));
    eprintln!("  🐢 slow: {}", stats.slow_path_hits.load(Ordering::Relaxed));

    if Path::new(&cli.checkpoint).exists() { let _ = fs::remove_file(&cli.checkpoint); }
    if cli.input.is_none() { let _ = fs::remove_file("_ocr_temp.jsonl"); }
    Ok(())
}
