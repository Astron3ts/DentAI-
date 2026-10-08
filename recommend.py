# recommend.py
"""
Clinical guidance and non-clinical screening index rules for Dent-AI.
Compliant with ethical safety protocols: never prescribes medicine or offers confirmed diagnoses.
"""

GUIDANCE_ROADMAP = {
    'Calculus': {
        'risk': 'Moderate',
        'severity_weight': 50,
        'summary': 'Hardened plaque deposit observed along the tooth margin.',
        'action': 'Schedule a professional ultrasonic dental scaling and prophylaxis.',
        'hygiene': 'Use tartar-control toothpaste and an electric toothbrush twice daily with daily flossing.',
        'referral_flag': 'Non-urgent dental appointment recommended within 2–4 weeks.'
    },
    'Caries': {
        'risk': 'High',
        'severity_weight': 80,
        'summary': 'Localized enamel demineralization or structural cavity formation observed.',
        'action': 'Consult a dentist promptly for restorative evaluation (filling, inlay, or crown).',
        'hygiene': 'Apply high-fluoride toothpaste twice daily, avoid sugary foods/drinks, and avoid chewing on the affected tooth.',
        'referral_flag': 'High priority dental examination recommended to prevent pulp involvement.'
    },
    'Gingivitis': {
        'risk': 'Moderate',
        'severity_weight': 45,
        'summary': 'Superficial inflammation and erythema along the gingival margin.',
        'action': 'Consult a dental hygienist for soft-tissue assessment and periodontal evaluation.',
        'hygiene': 'Rinse with warm saline or antiseptic chlorhexidine mouthwash (under dentist supervision); use ultra-soft bristled toothbrush.',
        'referral_flag': 'Routine dental consultation recommended within 2 weeks.'
    },
    'Mouth Ulcer': {
        'risk': 'Low to Moderate',
        'severity_weight': 35,
        'summary': 'Localized mucosal aphthous lesion with erythematous halo.',
        'action': 'Monitor lesion progression for 7–10 days. Seek clinical biopsy if persistent beyond 14 days.',
        'hygiene': 'Apply soothing topical anesthetic or protective oral paste; avoid acidic, spicy, or crunchy foods.',
        'referral_flag': 'Red-flag referral: Consult an oral medicine specialist if lesion persists > 2 weeks.'
    },
    'Tooth Discoloration': {
        'risk': 'Low',
        'severity_weight': 20,
        'summary': 'Extrinsic chromogenic staining or intrinsic shade alteration on tooth enamel.',
        'action': 'Schedule a dental checkup to rule out underlying decay or pulp necrosis.',
        'hygiene': 'Limit consumption of chromogenic beverages (coffee, tea, red wine); maintain consistent oral hygiene.',
        'referral_flag': 'Elective cosmetic consultation.'
    },
    'hypodontia': {
        'risk': 'Moderate',
        'severity_weight': 40,
        'summary': 'Developmental absence or premature loss of one or more dental units.',
        'action': 'Consult an orthodontist or prosthodontist for functional occlusion assessment.',
        'hygiene': 'Maintain space maintainers or adjacent teeth using specialized interdental brushes.',
        'referral_flag': 'Orthodontic / prosthodontic evaluation recommended.'
    }
}

RISK_ICONS = {
    'High': '🔴',
    'Moderate': '🟡',
    'Low to Moderate': '🟠',
    'Low': '🟢',
}

def screening_index(confidence_pct: float, predicted_class: str) -> int:
    """
    Computes a non-clinical screening index from 0 to 100 based on model confidence
    and condition severity weight.
    """
    base_weight = GUIDANCE_ROADMAP.get(predicted_class, {}).get('severity_weight', 40)
    # Scaled index: combines confidence (reliability) with clinical severity weighting
    index = int((confidence_pct / 100.0) * base_weight + (confidence_pct / 100.0) * 20)
    return max(5, min(100, index))

def generate_guidance(predicted_class: str) -> dict:
    """Returns the conservative clinical guidance dictionary for a given oral condition."""
    return GUIDANCE_ROADMAP.get(predicted_class, {
        'risk': 'Unknown',
        'summary': 'Unclassified condition pattern.',
        'action': 'Consult a licensed dental practitioner for an in-person clinical exam.',
        'hygiene': 'Maintain standard twice-daily brushing and flossing.',
        'referral_flag': 'Consult a licensed dental practitioner.'
    })
