use super::*;

/// Parses a Hugging Face repository reference (`org/repo`, `org/repo@rev`,
/// `hf://org/repo[@rev]`, or `https://huggingface.co/org/repo`) into `(repo_id, revision)`.
pub fn parse_repo_and_revision(
    input: &str,
    rev_override: Option<&str>,
) -> Result<(String, String)> {
    let trimmed = input.trim();
    let stripped = trimmed
        .strip_prefix(HF_PREFIX)
        .or_else(|| trimmed.strip_prefix("https://huggingface.co/"))
        .or_else(|| trimmed.strip_prefix("http://huggingface.co/"))
        .unwrap_or(trimmed)
        .trim_matches('/');

    let (body, at_rev) = match stripped.rsplit_once('@') {
        Some((b, r)) if !r.is_empty() && !r.contains('/') => (b, Some(r.to_string())),
        _ => (stripped, None),
    };

    let mut parts = body.split('/');
    let org = parts.next().unwrap_or("");
    let repo = parts.next().unwrap_or("");
    if org.is_empty() || repo.is_empty() || org.contains("..") || repo.contains("..") {
        bail!("invalid Hugging Face repo '{input}' (expected org/repo)");
    }
    let rev = rev_override
        .filter(|s| !s.trim().is_empty())
        .map(str::to_string)
        .or(at_rev)
        .unwrap_or_else(|| "main".to_string());
    Ok((format!("{org}/{repo}"), rev))
}

#[derive(Debug, Clone, Deserialize)]
pub struct HfTreeEntry {
    #[serde(rename = "type")]
    pub entry_type: Option<String>,
    pub path: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub lfs: Option<HfLfsInfo>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HfLfsInfo {
    #[serde(default)]
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct HfGgufCandidate {
    pub primary_file: String,
    pub quant: String,
    pub total_bytes: u64,
    pub shard_count: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuantRecommendation {
    pub candidate: HfGgufCandidate,
    pub hf_uri: String,
    pub verdict: String,
    pub pinned_pct: u8,
    pub est_load_s: f64,
    pub est_tps_low: f64,
    pub est_tps_high: f64,
    pub recommended: bool,
    pub score: i32,
}

fn extract_quant_label(path: &str) -> String {
    let fname = path.rsplit('/').next().unwrap_or(path);
    let stem = fname.strip_suffix(".gguf").unwrap_or(fname);
    // Strip trailing shard suffix `-00001-of-0000N` before scanning tokens.
    let clean = strip_shard_suffix(stem).map(|(p, _, _)| p).unwrap_or(stem);
    let upper = clean.to_ascii_uppercase();
    // Match well-known GGUF quant identifiers from most specific to least specific.
    const KNOWN: &[&str] = &[
        "UD-IQ1_S",
        "UD-IQ1_M",
        "UD-IQ2_XXS",
        "UD-IQ2_XS",
        "UD-IQ2_M",
        "UD-IQ3_XXS",
        "UD-Q2_K_XL",
        "UD-Q3_K_XL",
        "UD-Q4_K_XL",
        "UD-Q5_K_XL",
        "UD-Q6_K_XL",
        "UD-Q8_K_XL",
        "IQ1_S",
        "IQ1_M",
        "IQ2_XXS",
        "IQ2_XS",
        "IQ2_S",
        "IQ2_M",
        "IQ3_XXS",
        "IQ3_XS",
        "IQ3_S",
        "IQ3_M",
        "IQ4_XS",
        "IQ4_NL",
        "Q2_K_XL",
        "Q2_K_L",
        "Q2_K_S",
        "Q2_K",
        "Q3_K_XL",
        "Q3_K_L",
        "Q3_K_M",
        "Q3_K_S",
        "Q4_K_XL",
        "Q4_K_L",
        "Q4_K_M",
        "Q4_K_S",
        "Q4_1",
        "Q4_0",
        "Q5_K_XL",
        "Q5_K_L",
        "Q5_K_M",
        "Q5_K_S",
        "Q5_1",
        "Q5_0",
        "Q6_K_XL",
        "Q6_K_L",
        "Q6_K",
        "Q8_K_XL",
        "Q8_0",
        "MXFP4",
        "BF16",
        "FP16",
        "F16",
        "F32",
    ];
    for q in KNOWN {
        if upper.contains(q) {
            return (*q).to_string();
        }
    }
    clean
        .rsplit(['-', '_', '.'])
        .next()
        .unwrap_or("GGUF")
        .to_ascii_uppercase()
}

fn strip_shard_suffix(stem: &str) -> Option<(&str, u32, u32)> {
    let (prefix_and_idx, total_str) = stem.rsplit_once("-of-")?;
    if total_str.len() != 5 {
        return None;
    }
    let total: u32 = total_str.parse().ok()?;
    if total <= 1 {
        return None;
    }
    let (prefix, idx_str) = prefix_and_idx.rsplit_once('-')?;
    if idx_str.len() != 5 {
        return None;
    }
    let idx: u32 = idx_str.parse().ok()?;
    Some((prefix, idx, total))
}

fn quant_quality_rank(quant: &str) -> i32 {
    let (ud_bonus, base) = match quant.strip_prefix("UD-") {
        Some(rest) => (3, rest),
        None => (0, quant),
    };
    let rank = match base {
        "Q4_K_M" | "Q4_K_XL" => 100,
        "Q5_K_M" | "Q5_K_L" | "Q5_K_XL" => 98,
        "IQ4_XS" | "IQ4_NL" | "Q4_K_S" | "Q4_K_L" => 95,
        "Q6_K" | "Q6_K_L" | "Q6_K_XL" => 94,
        "Q8_0" | "Q8_K_XL" => 92,
        "Q5_K_S" | "Q5_0" | "Q5_1" | "Q4_0" | "Q4_1" => 88,
        "Q3_K_M" | "Q3_K_L" | "Q3_K_XL" => 80,
        "IQ3_M" | "IQ3_S" | "Q3_K_S" => 74,
        "IQ3_XS" | "IQ3_XXS" => 66,
        "Q2_K_XL" | "Q2_K_L" => 62,
        "Q2_K" | "Q2_K_S" | "IQ2_M" => 58,
        "BF16" | "FP16" | "F16" => 52, // huge footprint penalty on edge GPUs
        _ => 50,
    };
    rank + ud_bonus
}

/// Groups Hugging Face tree entries into logical GGUF checkpoints, combining multi-part shards
/// (`*-00001-of-0000N.gguf`) and filtering out `mmproj` vision projector files.
pub fn group_gguf_tree_entries(entries: &[HfTreeEntry]) -> Vec<HfGgufCandidate> {
    let mut grouped: BTreeMap<String, (String, u64, u32)> = BTreeMap::new();
    for entry in entries {
        if entry.entry_type.as_deref().is_some_and(|t| t != "file") {
            continue;
        }
        let lower = entry.path.to_ascii_lowercase();
        if !lower.ends_with(".gguf") || lower.contains("mmproj") {
            continue;
        }
        let bytes = entry
            .lfs
            .as_ref()
            .map(|l| l.size)
            .filter(|&s| s > 0)
            .unwrap_or(entry.size);
        let (dir_prefix, fname) = match entry.path.rsplit_once('/') {
            Some((d, f)) => (Some(d), f),
            None => (None, entry.path.as_str()),
        };
        let stem = fname.strip_suffix(".gguf").unwrap_or(fname);
        if let Some((prefix, idx, total)) = strip_shard_suffix(stem) {
            let key = match dir_prefix {
                Some(d) => format!("{d}/{prefix}"),
                None => prefix.to_string(),
            };
            let shard1_name = format!("{prefix}-00001-of-{total:05}.gguf");
            let primary = match dir_prefix {
                Some(d) => format!("{d}/{shard1_name}"),
                None => shard1_name,
            };
            let slot = grouped.entry(key).or_insert((primary, 0, total));
            slot.1 = slot.1.saturating_add(bytes);
            if idx == 1 {
                slot.0 = entry.path.clone();
            }
        } else {
            grouped.insert(entry.path.clone(), (entry.path.clone(), bytes, 1));
        }
    }

    let mut out: Vec<HfGgufCandidate> = grouped
        .into_values()
        .map(|(primary_file, total_bytes, shard_count)| HfGgufCandidate {
            quant: extract_quant_label(&primary_file),
            primary_file,
            total_bytes,
            shard_count,
        })
        .collect();
    out.sort_by_key(|c| c.total_bytes);
    out
}

/// Parses RFC 8288 `Link` header(s) and returns the `rel="next"` target URL, if present.
fn parse_next_link_url(headers: &ureq::http::HeaderMap, base_url: &str) -> Option<String> {
    for val in headers.get_all("link") {
        let Ok(raw) = val.to_str() else {
            continue;
        };
        for part in raw.split(',') {
            let mut segs = part.split(';').map(str::trim);
            let Some(url_part) = segs.next() else {
                continue;
            };
            let is_next = segs.any(|p| {
                p.eq_ignore_ascii_case("rel=\"next\"") || p.eq_ignore_ascii_case("rel=next")
            });
            if !is_next {
                continue;
            }
            let target = url_part
                .trim_start_matches('<')
                .trim_end_matches('>')
                .trim();
            if target.is_empty() {
                continue;
            }
            if target.starts_with("http://") || target.starts_with("https://") {
                return Some(target.to_string());
            }
            if target.starts_with('/')
                && let Some(origin) = url_origin(base_url)
            {
                return Some(format!("{origin}{target}"));
            }
        }
    }
    None
}

/// Fetches the recursive file tree of a Hugging Face repo (following `Link: rel="next"`
/// pagination on the configured Hub origin) and returns logical GGUF checkpoints.
pub fn list_hf_gguf_candidates(repo_id: &str, revision: &str) -> Result<Vec<HfGgufCandidate>> {
    let endpoint = std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_HF_ENDPOINT.to_string());
    let initial_url = format!(
        "{}/api/models/{}/tree/{}?recursive=true",
        endpoint.trim_end_matches('/'),
        repo_id,
        revision
    );
    let initial_origin = url_origin(&initial_url);

    let agent = ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_body(Some(Duration::from_secs(30)))
        .http_status_as_error(false)
        .build()
        .new_agent();

    let token = load_hf_token();
    let mut next_url = Some(initial_url);
    let mut all_entries: Vec<HfTreeEntry> = Vec::new();
    const MAX_TREE_PAGES: usize = 32;

    for _ in 0..MAX_TREE_PAGES {
        let Some(url) = next_url.take() else {
            break;
        };
        let mut req = agent.get(&url).header("User-Agent", "layerstream-plugin");
        if let Some(ref tok) = token
            && url_origin(&url) == initial_origin
        {
            req = req.header("Authorization", &format!("Bearer {tok}"));
        }

        let resp = req
            .call()
            .with_context(|| format!("query Hugging Face API {url}"))?;
        let status = resp.status().as_u16();
        if status != 200 {
            bail!("Hugging Face API returned HTTP {status} for {repo_id}@{revision} ({url})");
        }
        let candidate_next = parse_next_link_url(resp.headers(), &url);
        let body = resp
            .into_body()
            .read_to_string()
            .context("read Hugging Face tree JSON")?;
        let page_entries: Vec<HfTreeEntry> =
            serde_json::from_str(&body).context("parse Hugging Face tree JSON")?;
        all_entries.extend(page_entries);

        if let Some(nu) = candidate_next
            && url_origin(&nu) == initial_origin
        {
            next_url = Some(nu);
        }
    }
    Ok(group_gguf_tree_entries(&all_entries))
}

/// Evaluates and ranks GGUF candidates against probed (or user-specified) GPU/UMA memory
/// and sustained NVMe `O_DIRECT` bandwidth.
pub fn recommend_candidates(
    repo_id: &str,
    revision: &str,
    candidates: &[HfGgufCandidate],
    free_vram_bytes: u64,
    nvme_mib_s: f64,
    has_draft: bool,
) -> Vec<QuantRecommendation> {
    let free_gib = free_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let nvme_gib_s = (nvme_mib_s / 1024.0).max(0.05);
    let draft_reserve = if has_draft { 0.65 } else { 0.0 };
    let headroom = (free_gib * 0.14).max(1.0);

    let mut recs: Vec<QuantRecommendation> = candidates
        .iter()
        .map(|cand| {
            let model_gib = cand.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            let kv_and_slots = (model_gib * 0.08).clamp(0.8, 2.2);
            let usable = (free_gib - headroom - kv_and_slots - draft_reserve).max(0.0);
            let fits_resident = model_gib > 0.0 && model_gib <= usable;
            let pinned_frac = if fits_resident {
                1.0
            } else if model_gib > 0.0 {
                (usable / model_gib).clamp(0.0, 0.85)
            } else {
                0.0
            };
            let streamed_gib = (model_gib * (1.0 - pinned_frac)).max(0.0);
            let est_load_s = model_gib.min(usable + 0.8) / nvme_gib_s;

            let (verdict, est_tps_low, est_tps_high, fit_bonus) = if fits_resident {
                ("RESIDENT".to_string(), 15.0, 50.0, 200)
            } else if nvme_mib_s < 150.0 {
                ("TOO_SLOW_DISK".to_string(), 0.01, 0.05, -200)
            } else {
                let sweep_s = ((streamed_gib / nvme_gib_s) * 1.15).max(0.15);
                let mult_lo = if has_draft { 2.4 } else { 1.0 };
                let mult_hi = if has_draft { 5.2 } else { 1.0 };
                let lo = mult_lo / sweep_s;
                let hi = mult_hi / sweep_s;
                if lo >= 1.0 {
                    ("GOOD".to_string(), lo, hi, 140)
                } else if hi >= 0.5 {
                    ("SLOW".to_string(), lo, hi, 60)
                } else {
                    ("VERY_SLOW".to_string(), lo, hi, -40)
                }
            };

            let hf_uri = if revision == "main" {
                format!("hf://{repo_id}:{}", cand.primary_file)
            } else {
                format!("hf://{repo_id}:{}@{revision}", cand.primary_file)
            };

            let score = fit_bonus + quant_quality_rank(&cand.quant);
            QuantRecommendation {
                candidate: cand.clone(),
                hf_uri,
                verdict,
                pinned_pct: (pinned_frac * 100.0).round() as u8,
                est_load_s: (est_load_s * 10.0).round() / 10.0,
                est_tps_low: (est_tps_low * 100.0).round() / 100.0,
                est_tps_high: (est_tps_high * 100.0).round() / 100.0,
                recommended: false,
                score,
            }
        })
        .collect();

    if let Some((best_idx, _)) = recs
        .iter()
        .enumerate()
        .max_by_key(|(_, r)| (r.score, r.candidate.total_bytes))
    {
        recs[best_idx].recommended = true;
    }
    recs
}

/// Options controlling `resolve_pretrained_model` resolution and hardware-aware quant selection.
#[derive(Clone, Default)]
pub struct PretrainedResolveOptions {
    pub revision: Option<String>,
    pub quant: Option<String>,
    pub cache_dir: Option<PathBuf>,
    pub vram_budget_bytes: Option<u64>,
    pub use_gpu: bool,
    pub has_draft: bool,
    pub on_progress: Option<ProgressCallback>,
}

impl fmt::Debug for PretrainedResolveOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PretrainedResolveOptions")
            .field("revision", &self.revision)
            .field("quant", &self.quant)
            .field("cache_dir", &self.cache_dir)
            .field("vram_budget_bytes", &self.vram_budget_bytes)
            .field("use_gpu", &self.use_gpu)
            .field("has_draft", &self.has_draft)
            .field("has_on_progress", &self.on_progress.is_some())
            .finish()
    }
}

/// Result of resolving a local path, `hf://` URI, HF GGUF repo, or base HF repo to a local GGUF.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub local_path: PathBuf,
    pub source_uri: Option<String>,
    pub resolved_repo: Option<String>,
    pub selected_quant: Option<String>,
}

fn expand_tilde(path_str: &str) -> PathBuf {
    if let Some(rest) = path_str.strip_prefix("~/")
        && let Ok(home) = std::env::var("HOME")
        && !home.trim().is_empty()
    {
        return PathBuf::from(home).join(rest);
    }
    PathBuf::from(path_str)
}

fn looks_like_local_path(spec: &str) -> bool {
    let trimmed = spec.trim();
    if trimmed.starts_with(HF_PREFIX)
        || trimmed.starts_with("https://")
        || trimmed.starts_with("http://")
    {
        return false;
    }
    if trimmed.starts_with('/')
        || trimmed.starts_with("./")
        || trimmed.starts_with("../")
        || trimmed.starts_with("~/")
        || trimmed.contains('\\')
    {
        return true;
    }
    // Windows drive letter (e.g. `C:\...` or `C:/...`).
    let bytes = trimmed.as_bytes();
    if bytes.len() >= 3 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        return true;
    }
    trimmed.to_ascii_lowercase().ends_with(".gguf")
}

fn select_candidate_for_repo<'a>(
    recs: &'a [QuantRecommendation],
    quant_filter: Option<&str>,
    repo_id: &str,
) -> Result<&'a QuantRecommendation> {
    if let Some(wanted) = quant_filter.map(str::trim).filter(|s| !s.is_empty()) {
        let wanted_upper = wanted.to_ascii_uppercase();
        if let Some(found) = recs.iter().find(|r| {
            r.candidate.primary_file.eq_ignore_ascii_case(wanted)
                || r.candidate
                    .primary_file
                    .rsplit('/')
                    .next()
                    .is_some_and(|f| f.eq_ignore_ascii_case(wanted))
                || r.candidate.quant.eq_ignore_ascii_case(&wanted_upper)
        }) {
            return Ok(found);
        }
        let available: Vec<&str> = recs.iter().map(|r| r.candidate.quant.as_str()).collect();
        bail!(
            "quant '{wanted}' not found in '{repo_id}' (available: {})",
            available.join(", ")
        );
    }
    recs.iter()
        .find(|r| r.recommended)
        .or_else(|| recs.first())
        .with_context(|| format!("no GGUF candidates available in '{repo_id}'"))
}

/// Resolves a model specification into a local `.gguf` file path, downloading if needed.
///
/// Supported inputs:
/// 1. Local `.gguf` file path (`/path/to/model.gguf`, `./model.gguf`, `~/models/m.gguf`)
/// 2. Explicit `hf://org/repo:file.gguf[@rev]` or `hf://org/repo/file.gguf[@rev]` URI
/// 3. HF GGUF repository (`org/repo-GGUF[@rev]`, `hf://org/repo-GGUF`, HF URL) — picks `opts.quant`
///    or auto-selects the hardware-recommended quant for probed GPU/VRAM + NVMe bandwidth
/// 4. Base HF repository (`org/Model`) — when the repo contains no `.gguf` files, checks the
///    official `{org}/{Model}-GGUF` sibling repository and otherwise returns actionable guidance.
pub fn resolve_pretrained_model(
    spec: &str,
    opts: &PretrainedResolveOptions,
) -> Result<ResolvedModel> {
    resolve_pretrained_model_with_probe(spec, opts, || None)
}

/// Same as [`resolve_pretrained_model`], accepting a caller-supplied GPU free-VRAM probe
/// (returning free bytes) when `opts.use_gpu` is true and `opts.vram_budget_bytes` is unset.
pub fn resolve_pretrained_model_with_probe<F>(
    spec: &str,
    opts: &PretrainedResolveOptions,
    probe_free_vram: F,
) -> Result<ResolvedModel>
where
    F: FnOnce() -> Option<u64>,
{
    let trimmed = spec.trim();
    if trimmed.is_empty() {
        bail!("model specification must not be empty");
    }

    // 1. Local file path.
    let expanded = expand_tilde(trimmed);
    if expanded.is_file() {
        return Ok(ResolvedModel {
            local_path: expanded,
            source_uri: None,
            resolved_repo: None,
            selected_quant: None,
        });
    }
    if looks_like_local_path(trimmed) {
        bail!("local GGUF model not found: '{}'", expanded.display());
    }

    let cache_root = opts.cache_dir.clone().unwrap_or_else(default_cache_dir);

    // 2. Explicit `hf://org/repo:file.gguf` or `hf://org/repo/file.gguf` URI.
    if trimmed.starts_with(HF_PREFIX)
        && let Ok(mut parsed) = HfSpec::parse(trimmed)
    {
        if let Some(rev) = opts.revision.as_deref().filter(|s| !s.trim().is_empty())
            && !trimmed.contains('@')
        {
            parsed.revision = rev.to_string();
        }
        let uri = if parsed.revision == "main" {
            format!("hf://{}:{}", parsed.repo_id, parsed.filename)
        } else {
            format!(
                "hf://{}:{}@{}",
                parsed.repo_id, parsed.filename, parsed.revision
            )
        };
        let local_path =
            ensure_hf_downloaded_with_progress(&uri, &cache_root, opts.on_progress.as_ref())?;
        let selected_quant = Some(extract_quant_label(&parsed.filename));
        return Ok(ResolvedModel {
            local_path,
            source_uri: Some(uri),
            resolved_repo: Some(parsed.repo_id),
            selected_quant,
        });
    }

    // 3. HF repo ID (or base HF repo -> official `{org}/{name}-GGUF` sibling).
    let (repo_id, revision) = parse_repo_and_revision(trimmed, opts.revision.as_deref())?;
    let mut effective_repo = repo_id.clone();
    let mut candidates = list_hf_gguf_candidates(&repo_id, &revision)?;

    if candidates.is_empty() {
        if !repo_id.to_ascii_lowercase().ends_with("-gguf") {
            let sibling = format!("{repo_id}-GGUF");
            let model_name = repo_id.rsplit('/').next().unwrap_or(&repo_id);
            match list_hf_gguf_candidates(&sibling, &revision) {
                Ok(sib_cands) if !sib_cands.is_empty() => {
                    tracing::info!(
                        base_repo = %repo_id,
                        gguf_repo = %sibling,
                        candidates = sib_cands.len(),
                        "resolved base Hugging Face repository to official -GGUF sibling"
                    );
                    effective_repo = sibling;
                    candidates = sib_cands;
                }
                _ => {
                    bail!(
                        "repository '{repo_id}@{revision}' contains no .gguf files, and official \
                         sibling '{sibling}@{revision}' has no GGUF checkpoints. Pass an explicit \
                         GGUF repository (e.g. 'bartowski/{model_name}-GGUF' or \
                         'unsloth/{model_name}-GGUF'), an 'hf://org/repo:file.gguf' URI, or a \
                         local .gguf path"
                    );
                }
            }
        } else {
            bail!("no .gguf files found in Hugging Face repository '{repo_id}@{revision}'");
        }
    }

    let free_vram_bytes = opts
        .vram_budget_bytes
        .filter(|&b| b > 0)
        .or_else(|| {
            if opts.use_gpu {
                probe_free_vram()
            } else {
                None
            }
        })
        .unwrap_or(4 * 1024 * 1024 * 1024);

    let _ = fs::create_dir_all(&cache_root);
    let storage_bw = gguf_direct_io::probe_dir_storage_mib_per_s(&cache_root).unwrap_or(1500.0);

    let recs = recommend_candidates(
        &effective_repo,
        &revision,
        &candidates,
        free_vram_bytes,
        storage_bw,
        opts.has_draft,
    );
    let chosen = select_candidate_for_repo(&recs, opts.quant.as_deref(), &effective_repo)?;
    let local_path =
        ensure_hf_downloaded_with_progress(&chosen.hf_uri, &cache_root, opts.on_progress.as_ref())?;

    Ok(ResolvedModel {
        local_path,
        source_uri: Some(chosen.hf_uri.clone()),
        resolved_repo: Some(effective_repo),
        selected_quant: Some(chosen.candidate.quant.clone()),
    })
}

/// Maps a target GGUF `(architecture, model_hint)` to a known tiny (`<= 1B`) family-matched
/// Hugging Face GGUF draft repository for `draft_model="auto"`.
pub fn select_auto_draft_repo(architecture: &str, model_hint: &str) -> Option<&'static str> {
    let arch = architecture.trim().to_ascii_lowercase();
    let hint = model_hint.trim().to_ascii_lowercase();

    // Prioritize authoritative GGUF architecture keys before falling back to filename/repo hints.
    if arch.starts_with("qwen3") {
        return Some("Qwen/Qwen3-0.6B-GGUF");
    }
    if arch == "qwen2" || arch == "qwen2moe" {
        return Some("Qwen/Qwen2.5-0.5B-Instruct-GGUF");
    }
    if arch.starts_with("gemma3") {
        return Some("ggml-org/gemma-3-1b-it-GGUF");
    }
    if arch == "llama"
        && (hint.contains("llama-3")
            || hint.contains("llama3")
            || hint.contains("deepseek-r1-distill-llama"))
    {
        return Some("bartowski/Llama-3.2-1B-Instruct-GGUF");
    }
    if arch.is_empty() {
        if hint.contains("qwen3") {
            return Some("Qwen/Qwen3-0.6B-GGUF");
        }
        if hint.contains("qwen2.5") || hint.contains("qwen2-") || hint.contains("qwq") {
            return Some("Qwen/Qwen2.5-0.5B-Instruct-GGUF");
        }
        if hint.contains("gemma-3") || hint.contains("gemma3") {
            return Some("ggml-org/gemma-3-1b-it-GGUF");
        }
    }
    None
}

/// Resolves `draft_model="auto"` for an already-resolved target GGUF model:
/// - Skips small targets (`< 2 GiB`) and targets that already fit 100% resident in GPU VRAM.
/// - Skips when GPU VRAM headroom is too tight (`< 1.5 GiB` free).
/// - Selects a tiny family-matched GGUF draft model (`0.5B`–`1B`), downloads it into `cache_dir`,
///   and verifies `check_gguf_draft_compatible` against `target_path`.
/// - Returns `Ok(None)` cleanly on network/family/tokenizer misses (only propagating `on_progress`
///   user-cancellation errors) so `draft_model="auto"` never breaks offline or custom fine-tunes.
pub fn resolve_auto_draft_model(
    target: &ResolvedModel,
    cache_dir: Option<PathBuf>,
    vram_budget_bytes: Option<u64>,
    use_gpu: bool,
    on_progress: Option<ProgressCallback>,
) -> Result<Option<ResolvedModel>> {
    resolve_auto_draft_model_with_probe(
        target,
        cache_dir,
        vram_budget_bytes,
        use_gpu,
        on_progress,
        || None,
    )
}

/// Same as [`resolve_auto_draft_model`], accepting a caller-supplied GPU free-VRAM probe
/// (returning free bytes) when `use_gpu` is true and `vram_budget_bytes` is unset.
pub fn resolve_auto_draft_model_with_probe<F>(
    target: &ResolvedModel,
    cache_dir: Option<PathBuf>,
    vram_budget_bytes: Option<u64>,
    use_gpu: bool,
    on_progress: Option<ProgressCallback>,
    probe_free_vram: F,
) -> Result<Option<ResolvedModel>>
where
    F: FnOnce() -> Option<u64>,
{
    const MIN_AUTO_DRAFT_TARGET_BYTES: u64 = 2 * 1024 * 1024 * 1024;
    const MIN_AUTO_DRAFT_FREE_VRAM_BYTES: u64 = 1536 * 1024 * 1024;

    let Ok(idx) = gguf_direct_io::GgufIndex::open(&target.local_path) else {
        return Ok(None);
    };
    if idx.meta.total_file_bytes < MIN_AUTO_DRAFT_TARGET_BYTES {
        return Ok(None);
    }

    if use_gpu {
        let avail = vram_budget_bytes
            .filter(|&b| b > 0)
            .or_else(probe_free_vram);
        if let Some(free_b) = avail {
            if free_b < MIN_AUTO_DRAFT_FREE_VRAM_BYTES {
                return Ok(None);
            }
            // Target fits 100% resident in VRAM -> no out-of-core NVMe sweeps to amortize.
            if idx.meta.total_file_bytes.saturating_add(1024 * 1024 * 1024) < free_b {
                return Ok(None);
            }
        }
    }

    let hint = format!(
        "{} {} {}",
        idx.meta.name.as_deref().unwrap_or(""),
        target.resolved_repo.as_deref().unwrap_or(""),
        target
            .local_path
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("")
    );
    let Some(draft_repo) = select_auto_draft_repo(&idx.meta.architecture, &hint) else {
        tracing::info!(
            architecture = %idx.meta.architecture,
            hint = %hint.trim(),
            "draft_model='auto': no matching tiny draft family; using zero-VRAM n-gram speculation"
        );
        return Ok(None);
    };

    let cancelled_by_cb = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let wrapped_cb: Option<ProgressCallback> = on_progress.map(|cb| {
        let flag = cancelled_by_cb.clone();
        Arc::new(move |ev: &DownloadProgress| -> Result<()> {
            if let Err(e) = cb(ev) {
                flag.store(true, std::sync::atomic::Ordering::SeqCst);
                return Err(e);
            }
            Ok(())
        }) as ProgressCallback
    });

    let mut draft_opts = PretrainedResolveOptions {
        revision: None,
        quant: Some("Q4_K_M".to_string()),
        cache_dir,
        vram_budget_bytes: Some(2 * 1024 * 1024 * 1024),
        use_gpu,
        has_draft: false,
        on_progress: wrapped_cb,
    };

    let resolved = match resolve_pretrained_model(draft_repo, &draft_opts) {
        Ok(r) => r,
        Err(e) if cancelled_by_cb.load(std::sync::atomic::Ordering::SeqCst) => return Err(e),
        Err(_) => {
            draft_opts.quant = None;
            match resolve_pretrained_model(draft_repo, &draft_opts) {
                Ok(r) => r,
                Err(e) if cancelled_by_cb.load(std::sync::atomic::Ordering::SeqCst) => {
                    return Err(e);
                }
                Err(err) => {
                    tracing::info!(
                        draft_repo,
                        error = %err,
                        "draft_model='auto': could not fetch draft model; falling back to n-gram speculation"
                    );
                    return Ok(None);
                }
            }
        }
    };

    if let Err(reason) =
        gguf_direct_io::check_gguf_draft_compatible(&target.local_path, &resolved.local_path)
    {
        tracing::warn!(
            draft_repo,
            reason = %reason,
            "draft_model='auto': auto-selected draft tokenizer incompatible; falling back to n-gram speculation"
        );
        return Ok(None);
    }

    tracing::info!(
        draft_repo,
        path = %resolved.local_path.display(),
        "draft_model='auto': resolved compatible resident draft speculator"
    );
    Ok(Some(resolved))
}
