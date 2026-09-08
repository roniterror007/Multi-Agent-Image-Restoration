# Study Image Pipeline

## What this project does
This repository processes industrial spare-part images and standardizes them for ABB-style output.

Main goals:
- Segment object from cluttered backgrounds
- Correct exposure and color cast
- Keep object centered on a 1920x1440 white canvas
- Support human review with iterative fixes
- Learn per-image fixes via perceptual-hash memory

## Data reality check (from images folder)
The dataset is dominated by factory/spare-part photos on light or blue backgrounds, with recurring part families and lighting drift.

Decision based on this dataset:
- Keep the spare-part pipeline as the primary path
- Add a lightweight internal vision critic to detect degradation
- Do not prioritize face restoration workers (GFPGAN/CodeFormer) yet
- Treat monochrome inputs as a dedicated detail-and-lightness restoration case; the pipeline does not invent arbitrary object colors.

## Current architecture (runtime)

### Proposer path
1. Pre-flight heuristic checks brightness and neutrality
2. BiRefNet predicts mask
3. Exposure gate routes:
   - Path A (deterministic) for overexposure: Von Kries + CLAHE
   - Path B (neural residual): ColorMLP correction
4. Monochrome detection routes near-grayscale objects through stronger lightness recovery with reduced chroma/saturation movement
5. Optional blur repair for very low Laplacian variance
6. Optional pHash memory correction per image
7. Format to ABB canvas

### Color model and monochrome handling
ColorMLP predicts four residual controls: `L_Shift`, `A_Shift`, `B_Shift`, and `Sat_Gain`.

The current model has 16 image-level features:
- `V_P50`, entropy, Sobel edge density, white-background ratio
- Mean Lab `a`/`b`, mean saturation
- Laplacian variance in four quadrants and glare ratio
- Channel gap, saturation 95th percentile, low-saturation ratio, and luma interquartile range

The last four features improve recognition of grayscale and low-color inputs. During training, monochrome-like samples receive higher loss weight. At inference, monochrome routing can strengthen lightness correction while suppressing chroma and saturation changes. This restores usable contrast and detail without claiming to reconstruct unknown real-world colors.

### Domain-randomized training
`train_color_model.py` treats each image in `good_pics` as a clean target and synthesizes four reversible failure types per input:
- Warm, cool, pink, and green sensor/lighting casts from independent channel gains and offsets
- Underexposure, overexposure, and contrast loss
- Near-monochrome desaturation with retained luminance structure
- Smooth directional illumination and vignetting

The model learns the inverse global correction from the degraded feature vector back to the clean image statistics. Geometric transforms are deliberately not used for ColorMLP: it predicts global color controls, not geometry, masks, or pose. Segmentation robustness remains the responsibility of BiRefNet and the separate mask training flow.

`fusion_unet.pth` is currently a mask-refinement network, not a colorization model. Fully automatic color reconstruction for a genuinely grayscale object requires a separate colorization model trained with paired grayscale/color targets. A hint-guided U-Net is a suitable future model, but it must be trained and evaluated independently before its generated colors are used in production output.

### Critic path (new)
A dual-vote internal critic compares original vs corrected before finalizing:
- Quality vote: quality drop or major blur regression
- Consistency vote: excessive Lab mean shift or clipping ratio
- Severe failure override forces block
- Pre-review auto-remediation now tries candidate variants (weaker/stronger MLP, mask-recover path, monochrome-safe detail restore) and keeps the best non-degraded candidate.

If degraded:
- Output is fail-safe rolled back to original image for that item
- Item is marked for review in problems mode
- Critic metadata is logged in feedback records

### Human review path
- Tkinter review UI supports edit box and reference box
- Rejections trigger local fixes, pure-math correction, or VLM fallback
- Feedback is logged as JSONL and overrides are persisted by pHash
- Issue menu includes: color casts, brightness, saturation, low contrast, blur, monochrome detail loss, halo/artifact, bad mask, and other.

## Important architecture fixes applied
- Review corrections are now applied in corrected-image coordinates (not final canvas coordinates), then canvas is regenerated.
- This prevents mask mismatch, tiny pasted objects, and apparent fallback to old image.
- VLM fallback loading no longer offloads BiRefNet to CPU in a way that causes CUDA/CPU tensor mismatch.
- VLM JSON handling is hardened with extraction and retry behavior in controller logic.

## File map: what is actively used

### Runtime core
- `batch_process.py`: main pipeline, routing, critic, review loop, saving
- `agent_controller.py`: proportional controller, pure-math solver, VLM fallback control
- `review_ui.py`: interactive annotation and issue selection UI
- `feedback_store.py`: JSONL logging and pHash override storage/load
- `reviewer_agent.py`: applies per-image learned memory

### Models and runtime artifacts
- `color_predictor.pkl`: base 16-feature ColorMLP bundle used by default
- `color_predictor_feedback.pkl`: optional feedback-tuned bundle; auto-selected only when it contains at least 20 usable rejected feedback rows
- `models/llama-3.2-3b-instruct-q4_k_m.gguf`: local VLM fallback model (optional)
- `fusion_unet.pth`, `safety_valve.pth`: model artifacts not on the critical runtime path of `batch_process.py`

### Training and adaptation
- `train_color_model.py`: trains the base 16-feature model from `good_pics` with synthetic color degradations and monochrome-aware weighting
- `train_from_feedback.py`: fine-tunes from rejected feedback rows; legacy shorter vectors and NaN values are sanitized before training
- `apply_colour_model.py`: batch color enhancement helper using the same shared runtime feature extractor and model behavior as `batch_process.py`
- `train_fusion_unet.py`, `train_safety_valve.py`, `train_sope_model.py`: additional training scripts

### Retrieval and colorization
- `reference_retrieval.py`: builds structural descriptor index from reference library; stages monochrome candidates with ranked matches
- `reference_approval.py`: loads staged candidates; supports manual or threshold-based approval; applies approved color priors (Lab shifts)
- Tests: `tests/test_reference_retrieval.py`, `tests/test_reference_approval.py`

### Evaluation and support
- `quality_evaluator.py`: quality-related utilities
- `parameter_discovery_engine.py`: parameter search/experimentation
- `reference_inteligence.py`, `reference_report.json`: reference analysis artifacts
- `tests/`: regression and behavior tests

### Reference-based colorization workflow (new)
- `reference_retrieval.py`: builds retrieval index from `good_pics` reference library; scans `images/` to find monochrome candidates; stages them with ranked structural matches for human review.
- `reference_approval.py`: loads staged candidates, supports manual or threshold-based auto-approval, applies approved color priors to colorize candidates.
- `color_reference_workspace/`: generated workspace containing:
  - `reference_catalog.json`: metadata for 651+ reference images (Lab color priors)
  - `reference_index.npz`: compressed structural descriptors for fast similarity ranking
  - `color_reference_candidates.json`: 100 staged monochrome candidates, each with top 3 ranked references and similarity scores
  - `grayscale_candidates/`: staged monochrome images from `images/` folder
  - `reference_previews/`: visual preview pairs (candidate vs. reference matches)
  - `approvals*.jsonl`: approval decisions (approved/rejected per candidate)
  - `colorized_*/`: output directories with applied color priors (Lab shifts)

### Architectural Improvements (NEW)
See [ARCHITECTURAL_IMPROVEMENTS.md](ARCHITECTURAL_IMPROVEMENTS.md) for detailed documentation. Core improvements:

1. **Proposal-and-Verify Architecture** (replaces 3-step cascade routing)
   - All correction sources (pHash, Spatial Fields, VLM) generate proposals
   - Compete through shared quality gate: gamut safety, smoothness, perceptual improvement, coverage
   - Dynamic best-proposal selection (no hard routing)

2. **Spatial Correction Fields** (replaces global Lab shifts)
   - U-Net predicts [H, W, 3] per-pixel Lab shifts + [H, W] uncertainty maps
   - Handles mixed lighting scenarios (shadows vs highlights)
   - Gamut penalty integrated into loss (differentiable, not post-hoc)

3. **Trust-Region Line Search** (replaces fixed damping)
   - Mathematically rigorous convergence with Armijo condition
   - Adaptive trust-region radius (expands on success, contracts on failure)
   - Guaranteed monotonic objective improvement

4. **BiRefNet Tensor Isolation** (prevents rectangular box hallucination)
   - Clean and visualization tensors kept completely separate
   - BiRefNet receives only original image (no edit boxes in tensor)
   - Fallback segmentation via Canny edges or Otsu threshold

5. **LayerNorm in ColorMLP** (prevents NaN crashes)
   - Replaced no normalization with LayerNorm for numerical stability
   - Safe for single-sample inference
   - Prevents gradient explosion on batch size 1

6. **VLM Reasoning Scratchpad** (improves interpretability)
   - Optional `reasoning` field in VLM output
   - Model can "think" before outputting strict JSON ticks
   - Aids debugging and audit trails

### New modules implementing improvements
- `proposal_and_verify.py`: Proposal-and-Verify engine, SharedQualityGate
- `spatial_correction_fields.py`: SpatialCorrectionFieldGenerator, loss with gamut penalty
- `trust_region_line_search.py`: TrustRegionLineSearch, AdaptiveTrustRegionController
- `birefnet_tensor_isolation.py`: IsolatedBiRefNetProcessor, tensor validation, fallback recovery

### Data and outputs
- `images/`: source images
- `abb_output*/`: processed outputs
- `feedback*/`: annotations, fixed/isolated crops, feedback logs, pHash overrides
- `masks_*`: mask datasets and experiments
- `color_reference_workspace/`: retrieval staging and colorization workspace

## Feedback JSON fields you can rely on
Each record in `feedback.jsonl` contains route/review information plus critic telemetry.

Common fields:
- `applied_method` (method that produced the currently reviewed cleaned preview)

Critic fields (new):
- `critic_degraded`
- `critic_confidence`
- `critic_reasons`
- `critic_quality_drop`
- `critic_lab_shift`
- `critic_clip_ratio`

## How to run

### Monochrome reference-based colorization workflow
For images detected as monochrome (grayscale), the system can use structural similarity matching against a library of reference color images to apply learned color priors. This is a human-in-the-loop workflow that stages candidates for review before applying colors.

**Step 1: Build retrieval index and stage candidates**
```powershell
.\.venv\Scripts\python.exe .\reference_retrieval.py --references .\good_pics --input-dir .\images --output-dir .\color_reference_workspace --limit 100
```
This indexes 651 reference images from `good_pics`, scans `images/` for monochrome candidates (saturation < 14%), ranks each against all references using Sobel gradient descriptors, and stages the top 100 with their 3 best matches. Outputs:
- `color_reference_workspace/color_reference_candidates.json`: 100 staged candidates with recommended references
- `color_reference_workspace/grayscale_candidates/`: staged images
- `color_reference_workspace/reference_previews/`: visual review pairs

**Step 2: Approve candidates and apply colors**
High-confidence approval (similarity ≥ 0.95):
```powershell
.\.venv\Scripts\python.exe .\reference_approval.py --workspace .\color_reference_workspace --auto-approve-threshold 0.95 --approval-log .\color_reference_workspace/approvals.jsonl --apply-colors --output-dir .\color_reference_workspace/colorized_output
```
This auto-approves 80+ high-confidence matches and applies Lab color shifts to generate colorized versions. All edge cases (missing images, invalid JSON, corrupted files) are logged without silent failures.

Manual review (future UI):
- User inspects candidate/reference pairs in `reference_previews/`
- Approves or rejects each match via UI or file edit
- Approved matches logged to `approvals.jsonl`
- Color application runs only on approved candidates

### Standard reviewed run (full pipeline)

Each successful save prints ABB catalog metadata (from `images/DataSheet.xlsx`):
- `ABB Image ID`
- `Name`

### Fast non-interactive run
```powershell
.\.venv\Scripts\python.exe .\batch_process.py --input-dir .\images --output-dir .\abb_output_auto --feedback-dir .\feedback_auto --model .\color_predictor.pkl --review-mode none
```

### Default-model run
Omit `--model` to use `color_predictor.pkl`, or automatically use `color_predictor_feedback.pkl` after it has at least 20 usable feedback rows.
```powershell
.\.venv\Scripts\python.exe .\batch_process.py --input-dir .\images --output-dir .\abb_output_auto --feedback-dir .\feedback_auto --review-mode none
```

### Retrain the color model
Retrain the base bundle after changing features or expanding the clean-image dataset:
```powershell
.\.venv\Scripts\python.exe .\train_color_model.py
```

Fine-tune from reviewed feedback. Use a feedback directory with at least 20 rejected rows containing supported color/brightness issues before relying on the resulting bundle automatically.
```powershell
.\.venv\Scripts\python.exe .\train_from_feedback.py --feedback .\feedback_full\feedback.jsonl --feedback-dir .\feedback_full --model .\color_predictor.pkl --output .\color_predictor_feedback.pkl --epochs 120
```

### Monochrome smoke run
Run a non-interactive sample to verify monochrome routing and the final ABB canvases:
```powershell
.\.venv\Scripts\python.exe .\batch_process.py --input-dir .\images --output-dir .\abb_output_bw_smoke --feedback-dir .\feedback_bw_smoke --model .\color_predictor.pkl --review-mode none --limit 20
```

### Tests
Run all tests with comprehensive error reporting (no silent failures):
```powershell
.\.venv\Scripts\python.exe -m pytest tests -v --tb=short
```

Key test suites:
- `test_reference_retrieval.py`: Sobel descriptor computation, monochrome detection, ranking
- `test_reference_approval.py`: Lab shift application, approval recording, color application, error handling
- `test_cascade_router.py`, `test_color_math.py`, etc.: Core pipeline validation

All tests verify both success paths and failure cases (missing files, invalid data, type errors, JSON corruption, clipping, etc.).

Focused model-schema and regression validation:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_math_edge_cases.py tests/test_memory_stress.py tests/test_pipeline_e2e.py tests/test_picture_pipeline.py -q
```

### Failure and stress suites
```powershell
\.venv\Scripts\python.exe -m pytest tests/test_segmentation_vlm_robustness.py -q
\.venv\Scripts\python.exe -m pytest tests/test_memory_stress.py -q
```

### VLM schema hardening toggle
By default, extra JSON keys from VLM outputs are ignored if required keys are valid.

To reject any non-schema keys (production hardening mode):
```powershell
$env:STRICT_VLM_SCHEMA_KEYS="1"
\.venv\Scripts\python.exe .\batch_process.py --input-dir .\images --output-dir .\abb_output_strict --feedback-dir .\feedback_strict --model .\color_predictor.pkl --review-mode none
```

## Migration plan to DAG/microservices (pragmatic)
This repo can evolve to queues/workers, but do it in steps to avoid unnecessary complexity.

Phase 1 (now):
- Keep single-process pipeline
- Use critic to reduce blind outputs and reduce manual burden
- Keep feedback loop stable and auditable

Phase 2 (near term):
- Split into two scripts/processes:
  - Automated batch processor (segmentation + correction)
  - Async UAT processor (review queue + VLM)
- Store queue in SQLite first, then move to Redis/RabbitMQ if needed

Phase 3 (scale):
- Introduce event bus and dedicated workers
- Build DAG orchestration for intake, route, propose, critic, human-review, finalize
- Add confidence-based active-learning sampling

Phase 4 (specialized expansion):
- Add grayscale colorization worker only when grayscale volume justifies it
- Add continual-learning strategy (for example EWC) after stable retraining baselines

## Performance roadmap (ordered)
1. Data pipeline overlap (`pin_memory=True`, `non_blocking=True`) in batched loaders
2. Compile/inference optimization (`torch.compile`, then ONNX/TensorRT if ROI is proven)
3. Separate UAT lifecycle from automated run for VRAM isolation
4. Native extension only for confirmed hot loops after profiling

## Notes
- This project is optimized for spare-part restoration, not generic consumer photos.
- Avoid expanding architecture to unrelated domains until uncertainty metrics show recurring failure classes.
- The internal quality score and critic are safeguards, not ground-truth visual-quality metrics; review flagged outputs before using them as new training targets.
