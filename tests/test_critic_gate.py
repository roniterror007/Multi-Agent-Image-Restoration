import pytest
import numpy as np
import cv2
from xai.proposal_and_verify import ProposalEngine, CorrectionProposal, ProposalSource
from core.degradation_classifier import DegradationType, DegradationReport

def test_critic_gate_rejects_hallucinated_color():
    """
    Ensure the gate rejects color_mlp proposals that have extremely low confidence 
    due to out-of-distribution distance metrics.
    """
    engine = ProposalEngine()
    
    # Mock a safe but slightly less optimal VLM proposal
    vlm_proposal = CorrectionProposal(
        source_name=ProposalSource.VLM_REASONING,
        global_shifts=(0, 0, -15),
        confidence=0.85,
        reasoning="VLM detected yellow cast"
    )
    
    # Mock a ColorMLP proposal that is OOD (hallucinated) and has a decayed confidence of 0.15
    color_mlp_proposal = CorrectionProposal(
        source_name=ProposalSource.COLOR_MLP,
        global_shifts=(50, -50, 50), # Wild hallucination
        confidence=0.15,
        reasoning="ColorMLP predicted wild shift, but distance was 17.33 (OOD)."
    )
    
    dummy_lab = np.full((256, 256, 3), 128, dtype=np.float32)
    report = DegradationReport(primary=DegradationType.YELLOW_CAST, severity=0.8, secondary=[DegradationType.YELLOW_CAST], metrics={})
    
    best_proposal = engine.select_best(
        original_lab=dummy_lab,
        proposals=[vlm_proposal, color_mlp_proposal], 
        degradation=report
    )
    
    # The gate must select VLM because ColorMLP's confidence is too low
    assert best_proposal.source_name == ProposalSource.VLM_REASONING
    assert best_proposal.confidence == 0.85

def test_critic_gate_prefers_exemplar_with_high_sim():
    """
    Ensure the gate prefers exemplar transfer when the structural similarity is very high.
    """
    engine = ProposalEngine()
    
    # Dummy spatial lab image
    dummy_lab = np.full((256, 256, 3), 128, dtype=np.float32)
    dummy_lab_exemplar = np.full((256, 256, 3), 145, dtype=np.float32)
    
    exemplar_proposal = CorrectionProposal(
        source_name=ProposalSource.EXEMPLAR_TRANSFER,
        spatial_lab_image=dummy_lab_exemplar,
        confidence=0.95,
        ref_similarity=0.95,
        reasoning="High similarity DINOv2 match"
    )
    
    vlm_proposal = CorrectionProposal(
        source_name=ProposalSource.VLM_REASONING,
        global_shifts=(10, 0, 0),
        confidence=0.80,
        reasoning="VLM default"
    )
    
    report = DegradationReport(primary=DegradationType.LOW_CONTRAST, severity=0.8, secondary=[DegradationType.LOW_CONTRAST], metrics={})

    best_proposal = engine.select_best(
        original_lab=dummy_lab,
        proposals=[exemplar_proposal, vlm_proposal],
        degradation=report
    )
    
    # Exemplar should win due to high confidence and high reference similarity bonus
    assert best_proposal.source_name == ProposalSource.EXEMPLAR_TRANSFER

def test_critic_gate_spatial_fields_fallback():
    """
    Ensure spatial fields are selected if other models fail or have very low confidence.
    """
    engine = ProposalEngine()
    
    dummy_lab = np.full((256, 256, 3), 128, dtype=np.float32)
    dummy_lab_spatial = np.full((256, 256, 3), 145, dtype=np.float32)
    spatial_proposal = CorrectionProposal(
        source_name=ProposalSource.SPATIAL_FIELDS,
        spatial_lab_image=dummy_lab_spatial,
        confidence=0.60,
        reasoning="Spatial UNet Map"
    )
    
    color_mlp_proposal = CorrectionProposal(
        source_name=ProposalSource.COLOR_MLP,
        global_shifts=(10, 10, 10),
        confidence=0.30,
        reasoning="OOD"
    )
    
    report = DegradationReport(primary=DegradationType.GREEN_CAST, severity=0.8, secondary=[DegradationType.GREEN_CAST], metrics={})

    best_proposal = engine.select_best(
        original_lab=dummy_lab,
        proposals=[spatial_proposal, color_mlp_proposal],
        degradation=report
    )
    
    assert best_proposal.source_name == ProposalSource.SPATIAL_FIELDS
