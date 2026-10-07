"""Diagnosis: answer extraction, CoT checking, error taxonomy, LLTM, targeting."""

from .agreement import cohen_kappa, confusion_matrix, stratified_sample
from .align import Alignment, ReferenceTrace, RefStep, align
from .cot_parse import Equation, parse_equations, safe_eval
from .extract import answers_match, extract_answer, parse_number
from .lltm import LLTMResult, bootstrap_lltm, fit_lltm
from .targeting import (
    MatchReport, Weakness, baseline_threshold, choose_target, design_matrix, rank_weaknesses,
    select_matched_control, select_targeted, select_untargeted,
)
from .taxonomy import Diagnosis, ErrorType, classify, error_distribution

__all__ = [
    "cohen_kappa", "confusion_matrix", "stratified_sample", "Alignment", "ReferenceTrace", "RefStep",
    "align", "Equation", "parse_equations", "safe_eval", "answers_match", "extract_answer",
    "parse_number", "LLTMResult", "bootstrap_lltm", "fit_lltm", "MatchReport", "Weakness",
    "baseline_threshold", "choose_target", "design_matrix", "rank_weaknesses",
    "select_matched_control", "select_targeted", "select_untargeted", "Diagnosis", "ErrorType",
    "classify", "error_distribution",
]
