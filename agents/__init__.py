"""
agents/__init__.py

All imports are wrapped in try/except to prevent cascading failures
when some agent implementations are not yet available.
"""

agents = {}

# =============================================================================
# Core agents (CRL, GCFBC, GCIVL)
# =============================================================================
try:
    from agents.crl.original import CRLAgent
    agents['crl'] = CRLAgent
except Exception:
    pass

try:
    from agents.crl.byol import CRLBYOLAgent
    agents['crl_byol'] = CRLBYOLAgent
except Exception:
    pass

try:
    from agents.crl.dual import CRLDualAgent
    agents['crl_dual'] = CRLDualAgent
except Exception:
    pass

try:
    from agents.crl.tra import CRLTRAAgent
    agents['crl_tra'] = CRLTRAAgent
except Exception:
    pass

try:
    from agents.crl.vib import CRLVIBAgent
    agents['crl_vib'] = CRLVIBAgent
except Exception:
    pass

try:
    from agents.crl.vip import CRLVIPAgent
    agents['crl_vip'] = CRLVIPAgent
except Exception:
    pass

try:
    from agents.gcfbc.original import GCFBCAgent
    agents['gcfbc'] = GCFBCAgent
except Exception:
    pass

try:
    from agents.gcfbc.byol import GCFBCBYOLAgent
    agents['gcfbc_byol'] = GCFBCBYOLAgent
except Exception:
    pass

try:
    from agents.gcfbc.dual import GCFBCDualAgent
    agents['gcfbc_dual'] = GCFBCDualAgent
except Exception:
    pass

try:
    from agents.gcfbc.tra import GCFBCTRAAgent
    agents['gcfbc_tra'] = GCFBCTRAAgent
except Exception:
    pass

try:
    from agents.gcfbc.vib import GCFBCVIBAgent
    agents['gcfbc_vib'] = GCFBCVIBAgent
except Exception:
    pass

try:
    from agents.gcfbc.vip import GCFBCVIPAgent
    agents['gcfbc_vip'] = GCFBCVIPAgent
except Exception:
    pass

try:
    from agents.gcivl.original import GCIVLAgent
    agents['gcivl'] = GCIVLAgent
except Exception:
    pass

# =============================================================================
# GCIVL Pixel-based agents
# =============================================================================
try:
    from agents.gcivl.pixel.dual import GCIVLVisualDualAgent
    agents['gcivl_dual_vis'] = GCIVLVisualDualAgent
except Exception:
    pass

try:
    from agents.gcivl.pixel.byol import GCIVLVisualBYOLAgent
    agents['gcivl_byol_vis'] = GCIVLVisualBYOLAgent
except Exception:
    pass

try:
    from agents.gcivl.pixel.tra import GCIVLVisualTRAAgent
    agents['gcivl_tra_vis'] = GCIVLVisualTRAAgent
except Exception:
    pass

try:
    from agents.gcivl.pixel.vib import GCIVLVisualVIBAgent
    agents['gcivl_vib_vis'] = GCIVLVisualVIBAgent
except Exception:
    pass

try:
    from agents.gcivl.pixel.vip import GCIVLVisualVIPAgent
    agents['gcivl_vip_vis'] = GCIVLVisualVIPAgent
except Exception:
    pass

# =============================================================================
# GCIVL State-based agents
# =============================================================================
try:
    from agents.gcivl.state.dual import GCIVLDualAgent
    agents['gcivl_dual'] = GCIVLDualAgent
except Exception:
    pass

try:
    from agents.gcivl.state.byol import GCIVLBYOLAgent
    agents['gcivl_byol'] = GCIVLBYOLAgent
except Exception:
    pass

try:
    from agents.gcivl.state.tra import GCIVLTRAAgent
    agents['gcivl_tra'] = GCIVLTRAAgent
except Exception:
    pass

try:
    from agents.gcivl.state.vib import GCIVLVIBAgent
    agents['gcivl_vib'] = GCIVLVIBAgent
except Exception:
    pass

try:
    from agents.gcivl.state.vip import GCIVLVIPAgent
    agents['gcivl_vip'] = GCIVLVIPAgent
except Exception:
    pass

# =============================================================================
# Pixel-based cross-attention variants
# =============================================================================
try:
    from agents.gcivl.pixel.dual_crossattn import GCIVLVisualDualCrossAttnAgent
    agents['gcivl_dual_crossattn_vis'] = GCIVLVisualDualCrossAttnAgent
except Exception:
    pass

try:
    from agents.gcivl.pixel.dual_ms_crossattn import GCIVLVisualDualMSCrossAttnAgent
    agents['gcivl_dual_ms_crossattn_vis'] = GCIVLVisualDualMSCrossAttnAgent
except Exception:
    pass

try:
    from agents.gcivl.pixel.byol_crossattn import GCIVLVisualBYOLCrossAttnAgent
    agents['gcivl_byol_crossattn_vis'] = GCIVLVisualBYOLCrossAttnAgent
except Exception:
    pass

try:
    from agents.gcivl.pixel.tra_crossattn import GCIVLVisualTRACrossAttnAgent
    agents['gcivl_tra_crossattn_vis'] = GCIVLVisualTRACrossAttnAgent
except Exception:
    pass

try:
    from agents.gcivl.pixel.vib_crossattn import GCIVLVisualVIBCrossAttnAgent
    agents['gcivl_vib_crossattn_vis'] = GCIVLVisualVIBCrossAttnAgent
except Exception:
    pass

try:
    from agents.gcivl.pixel.vip_crossattn import GCIVLVisualVIPCrossAttnAgent
    agents['gcivl_vip_crossattn_vis'] = GCIVLVisualVIPCrossAttnAgent
except Exception:
    pass

# =============================================================================
# State-based cross-attention variants
# =============================================================================
try:
    from agents.gcivl.state.dual_crossattn import GCIVLDualCrossAttnAgent
    agents['gcivl_dual_crossattn'] = GCIVLDualCrossAttnAgent
except Exception:
    pass

try:
    from agents.gcivl.state.dual_ms_crossattn import GCIVLDualMSCrossAttnAgent
    agents['gcivl_dual_ms_crossattn'] = GCIVLDualMSCrossAttnAgent
except Exception:
    pass

try:
    from agents.gcivl.state.byol_crossattn import GCIVLBYOLCrossAttnAgent
    agents['gcivl_byol_crossattn'] = GCIVLBYOLCrossAttnAgent
except Exception:
    pass

try:
    from agents.gcivl.state.tra_crossattn import GCIVLTRACrossAttnAgent
    agents['gcivl_tra_crossattn'] = GCIVLTRACrossAttnAgent
except Exception:
    pass

try:
    from agents.gcivl.state.vib_crossattn import GCIVLVIBCrossAttnAgent
    agents['gcivl_vib_crossattn'] = GCIVLVIBCrossAttnAgent
except Exception:
    pass

try:
    from agents.gcivl.state.vip_crossattn import GCIVLVIPCrossAttnAgent
    agents['gcivl_vip_crossattn'] = GCIVLVIPCrossAttnAgent
except Exception:
    pass
