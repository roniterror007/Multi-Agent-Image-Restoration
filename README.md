# Antigravity Vision: Agent-Critic Continuous Learning Pipeline

An autonomous, multi-agent computer vision system designed to automatically detect, reason about, and repair severe degradations (color casts, over/underexposure, monochromaticity, and contrast issues) in industrial and photographic imagery. 

The system leverages a **Multi-Agent "Critic" Architecture** where multiple specialized models (DINOv2, BLIP VLM, Spatial UNets, and Color MLPs) generate competing restoration proposals. A mathematical "Critic Gate" then evaluates these proposals to select the safest, most accurate restoration.

---

## 🏗️ Architecture Overview

The pipeline operates in 6 distinct phases:

1. **Degradation Classification (`DegradationClassifier`)**: Analyzes the image statistically to detect if it's clean, B/W, has a color cast (pink, green, yellow, blue), or suffers from exposure/contrast issues.
2. **DINOv2 Semantic Retrieval**: Searches an exemplar database to find a visually and structurally identical healthy image to act as a reference.
3. **Multi-Agent Proposal Generation**:
   - **VLM Reasoner**: A HuggingFace `BLIP` model that describes the image and acts as a common-sense sanity check for color boundaries.
   - **Exemplar Transfer / Direct Replacement**: Uses DINOv2 structural matching combined with Cross-Attention/Sliced-Wasserstein algorithms to transfer colors from healthy references to the degraded image.
   - **Spatial Correction Fields**: A Dense U-Net predicting non-linear `Lab` shifts based on spatial coordinates.
   - **Color MLP**: A lightweight PyTorch Neural Network mapping global `a*b*` shifts.
4. **Critic Gate Debate (`ProposalEngine`)**: All agents submit their corrected images to the `SharedQualityGate`. The gate scores proposals based on structural similarity (SSIM), Gamut boundaries, Edge Loss, and Chroma Total Variation (TV). The highest-scoring proposal wins.
5. **Human-in-the-Loop Review UI**: A PyQt5 interactive UI that allows a human operator to review the AI's choices, manually override them via live sliders, and save the result.
6. **Continuous Learning (pHash Memory)**: Human overrides are saved into a JSON database keyed by the image's Perceptual Hash (pHash). If the pipeline encounters that image (or an identical crop) again, it instantly applies the human's exact historical fix.

---

## 📂 Project Structure & Folders

- **`core/`**: Core infrastructure modules.
  - `degradation_classifier.py`: Math-based heuristics and detection for specific visual degradations.
  - `vlm_reasoner.py`: The BLIP model integration for language-guided visual sanity checks.
  - `review_ui.py`: The PyQt5 Interactive Manual Review UI.
  - `feedback_store.py`: Perceptual Hashing (pHash) logic for the continuous learning database.
  - `agent_controller.py`: Proportional controllers and base agent classes.
- **`xai/`**: Explainable AI, Agent Logic, and Correction Field math.
  - `proposal_and_verify.py`: The core **Critic Gate** and `ProposalEngine` scoring math.
  - `birefnet_tensor_isolation.py`: Foreground extraction using BiRefNet.
  - `spatial_correction_fields.py`: Dense U-Net generation for spatial gradient maps.
- **`reference/`**: DINOv2 Retrieval & Transfer logic.
  - `exemplar_color_transfer.py`: Core logic for matching reference images and warping color histograms.
- **`tests/`**: Automated QA.
  - `test_critic_gate.py`: PyTest suite enforcing that the Critic Gate correctly rejects hallucinated or OOD (Out of Distribution) proposals.
- **`run_pipeline.py`**: The primary autonomous batch-processing loop.
- **`run_review.py`**: The human-in-the-loop manual review interface.
- **`images/`**: Put your input images here.
- **`abb_output_full/`**: Completed, corrected images are deposited here by the pipeline.
- **`feedback_auto/`**: The JSON memory bank where `pHash` overrides are persistently stored.

---

## 🚀 Setup & Installation

**Prerequisites:**
- Python 3.10+
- A CUDA-compatible GPU (e.g., RTX A2000, RTX 3090, or higher) for accelerated inference.

1. **Activate your environment:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
2. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
   *(Note: The environment already contains optimized versions of PyTorch, Transformers, PyQt5, and OpenCV).*

---

## ⚙️ Running the Pipeline

To run the autonomous processing pipeline across all images in your `images/` directory:

```powershell
python run_pipeline.py
```

**Optional Arguments:**
- `--input-dir <dir>`: Target folder for raw images (default: `images`).
- `--output-dir <dir>`: Target folder for processed images (default: `abb_output_full`).
- `--limit <int>`: Only process the first N images (useful for quick testing).

*Example:* `python run_pipeline.py --input-dir bad_images --limit 5`

---

## 🧑‍💻 Interactive Manual Review (Human-in-the-loop)

Once the pipeline has processed images into `abb_output_full/`, you can review them and inject your own human expertise back into the system:

```powershell
python run_review.py
```

**Using the UI:**
- **Hotkeys:**
  - `T`: Toggle between Dark Mode and Light Mode dynamically.
- **Sliders:** Moving the `Lightness (L)`, `Green-Red (a*)`, or `Blue-Yellow (b*)` sliders overrides the AI.
- **Accepting:** Clicking "Accept Current Preview" while using a slider saves your custom fix to `feedback_auto/image_overrides.json`.
- **Killing the Pipeline:** Escapes the loop and halts the process entirely.

---

## 🧠 Automated QA

The mathematical integrity of the Critic Gate is guarded by a suite of automated unit tests. To verify the system's robustness against hallucinated models or regressions, run:

```powershell
pytest tests/test_critic_gate.py
```
This ensures the gate continues to prefer safe, mathematically verified structural matches over low-confidence or heavily destructive corrections.
