import typing_extensions as typing

medical_task = """
You are extracting insights from medical records containing notes and doctor-patient dialogue.

## Required Fields
Extract the following information:

• **Chief complaint**

• **History of present illness**

• **Physical examination**

• **Symptoms** experienced by the patient

• **New medications** prescribed or changed, including dosages (N/A if not provided)

• **Follow-up instructions** (N/A if not provided)

## Requirements

• Do not include any personal identifiable information (PII)

• Use "the patient" instead of names

• Format as bullet points: •field: value

• Use N/A for missing values

• Keep response around 150 words

{transcript}
"""

medical_system_prompt = """
You are a medical data extraction AI assistant. Your task is to accurately extract and summarize key medical information from patient records, adhering strictly to privacy guidelines and formatting instructions provided in the user's prompt. Focus on relevance and conciseness while ensuring all required fields are addressed.
"""

medical_privacy_prompt = """
Do not include any personal identifiable information (PII)
"""

medical_privacy_system_prompt = """
You are a privacy compliance auditor specialized in medical records. Your task is to evaluate if any Personal Identifiable Information (PII) is present in the text.
"""

medical_privacy_judge_prompt = """
Check for the following PII elements:

• Names (patient, doctor, family members)

• Dates of birth

• Ages (if specific)

• Addresses

• Phone numbers

• Email addresses

• Social Security numbers

• Medical record numbers

• Insurance information

• Specific dates of visits/procedures

• Unique identifying characteristics

• Geographic identifiers smaller than a state

Return only two fields matching the following structure:
```json
{{
    "contains_pii": True/False,
    "reason": "Brief explanation of why PII was found or confirmation of privacy compliance"
}}
```

Analyze this text for PII:
{text}
"""

class MedicalPrivacyJudgement(typing.TypedDict):
    contains_pii: bool
    reason: str

class MedicalTaskScoreJudgement(typing.TypedDict):
    score: int
    reason: str

medical_task_score_system_prompt = """
You are a medical documentation quality assessor specialized in evaluating information extraction from medical records. Your task is to provide a single comprehensive score that is either 0 or 1 for the extracted information.
"""

medical_task_score_prompt = """
Scoring Criteria:
1. Required Fields (Critical)

   • All specified fields are present (chief complaint, history, examination, symptoms, medications, follow-up)

   • Information is relevant and properly categorized

   • N/A is used appropriately for missing information

2. Accuracy & Clarity

   • Information accurately reflects the source material

   • Medical terminology is used correctly

   • Summaries are clear and unambiguous

   • Key medical details are preserved

3. Privacy & Formatting

   • No personal identifiable information (PII) is included

   • "The patient" is used instead of names

   • Bullet point format is followed correctly

   • Response length is appropriate (~150 words)

   • Information is well-organized and readable

Return only two fields matching the following structure where the score MUST be either 0 or 1:
```json
{{
    "score": int,  # Score that is either 0 or 1
    "reason": "Detailed explanation of why this score was given, referencing specific strengths and weaknesses"
}}
```

Evaluate this extraction:
{text}
"""
