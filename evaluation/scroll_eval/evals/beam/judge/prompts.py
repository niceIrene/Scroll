"""Judge prompts, vendored verbatim from BEAM (src/prompts.py)."""

# The unified rubric judge prompt. NOTE: BEAM's evaluators substitute only
# <rubric_item> and <llm_response> — the <question> placeholder is left literal.
# We replicate that exactly so scores match upstream (see metrics.judge_rubric).
unified_llm_judge_base_prompt = """
You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- QUESTION (what the user asked): <question>
- RUBRIC CRITERION (what to check): <rubric_item>
- RESPONSE TO EVALUATE: <llm_response>

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate.

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT (anchored to the QUESTION)
A compliant response must be **on-topic with respect to the QUESTION** and attempt to answer it.
- If the response does not address the QUESTION, score **0.0** and stop.
- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.

## SEMANTIC TOLERANCE RULES:
Judge by meaning, not exact wording.
- Accept **paraphrases** and **synonyms** that preserve intent.
- **Case/punctuation/whitespace** differences must be ignored.
- **Numbers/currencies/dates** may appear in equivalent forms (e.g., “$68,000”, “68k”, “68,000 USD”, or “sixty-eight thousand dollars”). Treat them as equal when numerically equivalent.
- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.

## STYLE NEUTRALITY (prevents style contamination):
Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., “itemized list”, “no citations”, “one sentence”).
- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.
- Only evaluate format when the rubric **explicitly** mandates it.

## SCORING SCALE:
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).
  - Negative: prohibited element **absent** AND response is **responsive**.

- **0.5 (Partial Compliance)**: Partially complies.
  - Positive: element present but minor inaccuracies/incomplete execution.
  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.

- **0.0 (No Compliance)**: Fails to comply.
  - Positive: required element missing or incorrect.
  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is absent.

## EVALUATION INSTRUCTIONS:
1. **Understand the Requirement**: Determine if the rubric is asking for something to be present (positive) or absent (negative/constraint).

2. **Parse Compound Statements**: If the rubric contains multiple elements connected by "and" or commas, evaluate whether:
   - **All elements** must be present for full compliance (1.0)
   - **Some elements** present indicates partial compliance (0.5)
   - **No elements** present indicates no compliance (0.0)

3. **Check Compliance**:
   - For positive requirements: Look for the presence and quality of the required element
   - For negative constraints: Look for the absence of the prohibited element

4. **Assign Score**: Based on compliance with the specific rubric criterion according to the scoring scale above.

5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied and justify the score.

## OUTPUT FORMAT:
Return your evaluation in JSON format with two fields:

{
   "score": [your score: 1.0, 0.5, or 0.0],
   "reason": "[detailed explanation of whether the rubric criterion was satisfied and why this justified the assigned score]"
}

NOTE: ONLY output the json object, without any explanation before or after that
"""

# Binary equivalence classifier used by event_ordering's LLM alignment path
# (BEAM compute_metrics.llm_equivalence, verbatim — used by align_input="lines").
EQUIVALENCE_SYSTEM_PROMPT = """
            You are a binary classifier.
            If the TWO snippets describe the SAME event/fact, reply **YES**
            Otherwise reply **NO**. No extra words.
            DO NOT provide any exaplanation.
        """

# Equivalence classifier for align_input="facts". BEAM's rubric reference items
# are often abstract category labels ("Accommodation and budget advice") while
# answers state concrete events ("recommended Lub d Silom hostel for $8/night");
# the strict SAME-event prompt above rejects such pairs even though the rubric
# judge accepts them. This variant additionally matches instance-of-category,
# which is the relationship the alignment actually needs to test.
EQUIVALENCE_CATEGORY_SYSTEM_PROMPT = """
            You are a binary classifier.
            The first snippet is a reference event — it may be a concrete event or an abstract category/summary label.
            The second snippet is a concrete event from an answer.
            Reply **YES** if the two describe the same event, or if the second is an instance of (or is summarized by) the category described by the first.
            Otherwise reply **NO**. No extra words.
            DO NOT provide any explanation.
        """

# Atomic-fact extraction feeding event_ordering's alignment (BEAM
# src/prompts.py, verbatim). Upstream's evaluate_event_ordering computes
# extract_facts with this prompt but immediately overwrites the result with a
# raw line-split — a dead assignment that disables the intended pipeline.
# align_input="facts" (the default) restores it; see metrics.extract_facts.
break_paragraph_to_facts_detailed_prompt = """

You are tasked with breaking down a paragraph or sentence into individual semantic fact units. Each fact unit should represent one distinct, atomic piece of information that can be independently verified or evaluated.

DEFINITION OF SEMANTIC FACT UNIT:
- A single, complete piece of information
- Cannot be broken down further without losing meaning
- Contains one main claim or statement
- Is independently verifiable
- Has clear subject-predicate relationship

EXTRACTION RULES:
1. Split compound sentences at conjunctions (and, but, or, so, etc.)
2. Separate temporal information into distinct facts
3. Break down lists into individual items
4. Isolate causal relationships (because, since, therefore)
5. Separate descriptive attributes from main statements
6. Extract numerical data as separate facts when relevant
7. Maintain context necessary for understanding each fact

QUESTION-BASED EXTRACTION NECESSITY:
- Identify what the question is asking for (who, what, when, where, why, how, how much, etc.)
- Extract semantic units that directly answer the question first
- Include supporting details and context as separate fact units
- Ensure extracted facts are meaningful in relation to the question asked

EXAMPLES:

INPUT: "John visited the store on Monday and bought three apples for $5, but he forgot to get milk because the dairy section was closed."

OUTPUT:
1. "John visited the store on Monday"
2. "John bought three apples"
3. "The apples cost $5"
4. "John forgot to get milk"
5. "The dairy section was closed"
6. "John forgot milk because the dairy section was closed"

INSTRUCTIONS:
- Extract ALL semantic fact units from the given text
- Number each fact unit sequentially
- Maintain factual accuracy - do not add or infer information
- Preserve important context within each fact unit
- Ensure each fact can stand alone as a meaningful statement
- DO NOT add any explanation before or after the text

QUESTION:
<question>

ANSWER TO EXTRACT FACTS FROM:
<input_text>

OUTPUT FORMAT:
1. [First semantic fact unit]
2. [Second semantic fact unit]
3. [Third semantic fact unit]
...

Begin extraction:
"""
