import pandas as pd
from collections import defaultdict

ALL_PROMPTS = {

"zero_13": ("""You will be shown ONE document. Classify the political bias of THIS document only — ignore any other text, examples, or formatting that might appear elsewhere in this prompt.

Label and Definitions:
1 - LEFT: uses language, framing, or selective emphasis that favors progressive positions, Democratic figures, or criticizes conservative figures/policies more harshly than the reverse.
2 - RIGHT: uses language, framing, or selective emphasis that favors conservative positions, Republican figures, or criticizes progressive figures/policies more harshly than the reverse.
3 - NEUTRAL: reports facts, quotes, or events without loaded language or one-sided framing — even if the topic itself is political (e.g. a candidate announcing a run, a bill being signed, a court ruling). Most straight news reporting is NEUTRAL. Only choose 1 (LEFT) or 2 (RIGHT) if there is clear evidence of imbalance in THIS specific text (loaded adjectives, one-sided sourcing, editorializing verbs like "slams"/"disgraced"/"radical", omission of an opposing view on a contested claim).

Important: 
- The topic being political does NOT make the document biased. A factual document about a Republican or Democrat is NEUTRAL unless the language itself is slanted.

DOCUMENT:
{content}

Classification:
1. LEFT
2. RIGHT
3. NEUTRAL

Respond with only the digit (1, 2, or 3) and no explanation.
ANSWER:""", {1: 'left', 2: 'right', 3: 'neutral'}),

"zero_12": ("""You are an expert in political media bias analysis.

STEP 1 — Determine if the document is political at all.
If it has no meaningful connection to partisan politics, policy debates, elections,
or ideological issues (e.g. sports scores, recipes, product reviews, weather),
classify as 2 (NEUTRAL) immediately and skip to the ANSWER line.

STEP 2 — If political, evaluate these signals:
- Word choice: does it use framing more common in left-leaning outlets
  (e.g. "reproductive rights," "gun violence," "undocumented immigrants") or
  right-leaning outlets (e.g. "unborn child," "gun rights," "illegal aliens")?
- Topic selection & emphasis: which facts are foregrounded, which are omitted
  or downplayed?
- Source selection: are quoted experts/officials predominantly from one
  political side?
- Policy framing: is a policy described in terms favorable to progressive
  goals (equity, regulation, collective action) or conservative goals
  (individual liberty, free markets, tradition)?
- Tone toward political figures/groups: is criticism or praise distributed
  evenly across parties, or concentrated on one side?

STEP 3 — Decide.
Only classify as 2 (NEUTRAL) for a political document if it presents multiple
viewpoints fairly, uses neutral language for all sides, and does not
consistently favor one side across the signals above. Do not default to 2
just because the lean is subtle — weak-but-present bias should still be
classified as 1 or 3. Reserve 2 for genuinely balanced political coverage
or non-political content.

DOCUMENT:
{content}

Classification:
1. LEFT
2. NEUTRAL
3. RIGHT

Respond with only the digit (1, 2, or 3) and no explanation.
ANSWER:""", {1: 'left', 2: 'neutral', 3: 'right'}),

"zero_11": ("""You are an expert in political media bias. Classify the document below as one of:
1 = LEFT
2 = NEUTRAL
3 = RIGHT

Base your classification only on the document content, using these criteria:
- Language: use of partisan or loaded terms
- Position: alignment with progressive or conservative policies
- Framing: balanced coverage vs. one-sided presentation

If the document does not clearly lean left or right, answer 2.

DOCUMENT:
{content}

Classification:
1. LEFT
2. NEUTRAL
3. RIGHT

Respond with only the digit (1, 2, or 3) and no explanation.
ANSWER:""", {1: 'left', 2: 'neutral', 3: 'right'}),

# "zero_11_reasoning": ("""You are an expert in political media bias. Classify the document below as one of:
# 1 = LEFT
# 2 = NEUTRAL
# 3 = RIGHT

# Base your classification only on the document content, using these criteria:
# - Language: use of partisan or loaded terms
# - Position: alignment with progressive or conservative policies
# - Framing: balanced coverage vs. one-sided presentation

# If the document does not clearly lean left or right, answer 2.
# DOCUMENT:
# {content}

# First explain your reasoning in 2-3 sentences on a line starting with "REASONING:".
# Then, on a new line, respond with only the digit (1, 2, or 3) on a line starting with "ANSWER:".
# REASONING:""", {1: 'left', 2: 'neutral', 3: 'right'}),

# "zero_10": ("""You are an expert in political media bias.

# Classify the document below as one of:
# 1 = LEFT
# 2 = NEUTRAL
# 3 = RIGHT

# Use NEUTRAL (2) for either of these cases:
# - The document covers a political topic but presents it in a balanced, non-partisan way
# - The document is not political at all (e.g. sports, weather, entertainment, science, business/finance without policy angles, health, technology, etc.)

# Only classify as LEFT or RIGHT if the document is political in nature AND shows a clear partisan lean.

# Base your classification only on the document content, using these criteria:
# - Language: use of partisan or loaded terms
# - Position: alignment with progressive or conservative policies
# - Framing: balanced coverage vs. one-sided presentation

# DOCUMENT:
# {content}

# Classification:
# 1. LEFT
# 2. NEUTRAL
# 3. RIGHT

# Respond with only the digit (1, 2, or 3) and no explanation.
# ANSWER:""", {1: 'left', 2: 'neutral', 3: 'right'}),
 
"zero_9": ("""You are an expert in political media bias.

Classify the document below as one of:
1 = LEFT-leaning
2 = NEUTRAL
3 = RIGHT-leaning

Base your classification only on the document content, using these criteria:
- Language: use of partisan or loaded terms
- Position: alignment with progressive or conservative policies
- Framing: balanced coverage vs. one-sided presentation

DOCUMENT:
{content}

Classification:
1. LEFT
2. NEUTRAL
3. RIGHT

Respond with only the digit (1, 2, or 3) and no explanation.
ANSWER:""", {1: 'left', 2: 'neutral', 3: 'right'}),

    'zero_2': ("""
You are an expert in media bias detection. Read the document below and classify its political slant into one of three categories: Left, Neutral, or Right.

Consider the following criteria:
- Does the document use partisan framing or terminology (e.g., "pro-life" vs. "anti-choice")?
- Does the document endorse or criticize political views that correspond more with the Democratic or Republican platform?
- Does the document selectively emphasize or omit facts in ways that align with a partisan agenda?

Based on this analysis, classify the document as:
1. Left: Stronger alignment with Democratic viewpoints or framing
2. Neutral: Balanced, with minimal or no partisan indicators
3. Right: Stronger alignment with Republican viewpoints or framing

DOCUMENT:
{content}

Classification:
1. LEFT
2. NEUTRAL
3. RIGHT

Respond with only the digit (1, 2, or 3) and no explanation.
ANSWER:""", {1: 'left', 2: 'neutral', 3: 'right'}),

    'zero_3': ("""
You are a political media analyst. Your task is to classify the political bias of the following document into one of three categories: left, neutral, or right.

Follow these steps:
1. Language: Identify whether the document uses partisan language (e.g., "death tax" for right-leaning, "estate tax" for left-leaning).
2. Political Position: Determine whether the political positions align more with the Democratic or Republican Party, or are neutral.
3. Coverage: Assess whether the document selectively highlights issues prioritized by one party while ignoring the other.

Use this to guide your classification:
1. Left: Favors Democratic language, positions, and/or issue coverage.
2. Right: Favors Republican language, positions, and/or issue coverage.
3. Neutral: Balanced in language, positions, and issue coverage.

DOCUMENT:
{content}

Classification:
1. LEFT
2. RIGHT
3. NEUTRAL

Respond with only the digit (1, 2, or 3) and no explanation.
ANSWER:""", {1: 'left', 2: 'right', 3: 'neutral'}),

    'Zero-shot 1': ("""
You are an expert political analyst tasked with classifying documents by their political slant based on three key factors: language choices, political positions, and issue coverage. Please analyze the following document and classify its political slant as LEFT, NEUTRAL, or RIGHT.

Consider these three factors in your analysis:

Language Analysis: Look for partisan terminology
- Right-leaning terms: "death tax," "job creators," "government overreach," "pro-life"
- Left-leaning terms: "estate tax," "wealthy elite," "corporate welfare," "pro-choice"

Political Position: Assess alignment with party positions
- Right-leaning: supports conservative policies (lower taxes, deregulation, traditional values)
- Left-leaning: supports progressive policies (social programs, regulation, social justice)

Coverage Selection: Evaluate what aspects are emphasized or omitted
- Does the document present only one side of a multi-faceted issue?
- Are counterarguments or alternative perspectives acknowledged?

Document to analyze:
{content}

Classification: Based on your analysis, classify this document as:
1. LEFT: Shows clear bias toward Democratic/progressive viewpoints
2. NEUTRAL: Presents balanced coverage without clear partisan lean
3. RIGHT: Shows clear bias toward Republican/conservative viewpoints

Respond with only the digit (1, 2, or 3) and no explanation.
ANSWER:""", {1: 'left', 2: 'neutral', 3: 'right'}),

    'Zero-shot-1-right': ("""
You are an expert political analyst tasked with classifying documents by their political slant based on three key factors: language choices, political positions, and issue coverage. Please analyze the following document and classify its political slant as LEFT, NEUTRAL, or RIGHT.

Consider these three factors in your analysis:

Language Analysis: Look for partisan terminology
- Right-leaning terms: "death tax," "job creators," "government overreach," "pro-life"
- Left-leaning terms: "estate tax," "wealthy elite," "corporate welfare," "pro-choice"

Political Position: Assess alignment with party positions
- Right-leaning: supports conservative policies (lower taxes, deregulation, traditional values)
- Left-leaning: supports progressive policies (social programs, regulation, social justice)

Coverage Selection: Evaluate what aspects are emphasized or omitted
- Does the document present only one side of a multi-faceted issue?
- Are counterarguments or alternative perspectives acknowledged?

Document to analyze:
{content}

Classification: Based on your analysis, classify this document as:
1. LEFT: Shows clear bias toward Democratic/progressive viewpoints
2. NEUTRAL: Presents balanced coverage without clear partisan lean
3. RIGHT: Shows clear bias toward Republican/conservative viewpoints

Respond with only the digit (1, 2, or 3) and no explanation.
ANSWER:""", {1: 'left', 2: 'neutral', 3: 'right'}),

    'zero_5': ("""
You are a media bias analyst whose task is to evaluate documents for political slant using a systematic approach. Focus on objectivity and evidence-based classification. Analyze the following document for political bias and classify it as LEFT, NEUTRAL, or RIGHT.

Follow these steps:

Step 1: Language Examination
- Identify any politically charged terms or phrases
- Note if language favors Republican or Democratic framing of issues
- Examples: "illegal aliens" vs "undocumented immigrants," "climate change" vs "global warming"

Step 2: Position Assessment
- What political stance does the document take on key issues?
- Does it align with typical Republican positions (fiscal conservatism, traditional values, strong defense)?
- Does it align with typical Democratic positions (social programs, environmental protection, civil rights)?

Step 3: Coverage Balance
- What aspects of the story are highlighted vs. downplayed?
- Are opposing viewpoints given fair representation?
- Does the selection of facts favor one political perspective?

Step 4: Overall Classification
Based on your analysis above, classify the document as:
1. LEFT: Demonstrates bias toward progressive/Democratic perspectives
2. NEUTRAL: Shows balanced reporting without clear partisan lean
3. RIGHT: Demonstrates bias toward conservative/Republican perspectives

DOCUMENT:
{content}

Classification:
1. LEFT
2. NEUTRAL
3. RIGHT

Respond with only the digit (1, 2, or 3) and no explanation.
ANSWER:""", {1: 'left', 2: 'neutral', 3: 'right'}),

    'zero_6': ("""
You are trained to identify political bias in news reporting and should classify documents into three categories: LEFT, NEUTRAL, or RIGHT political slant. Classify the political slant of this document using the following framework.

Classification Criteria:

LEFT Classification - Document shows bias toward Democratic/progressive viewpoints through:
- Use of language preferred by Democrats (e.g., "reproductive rights," "climate crisis," "wealth inequality")
- Presenting positions that align with Democratic party platform
- Emphasizing issues important to Democratic voters while minimizing conservative concerns
- Framing stories in ways that support progressive policy goals

RIGHT Classification - Document shows bias toward Republican/conservative viewpoints through:
- Use of language preferred by Republicans (e.g., "right to life," "border security," "job creators")
- Presenting positions that align with Republican party platform
- Emphasizing issues important to Republican voters while minimizing progressive concerns
- Framing stories in ways that support conservative policy goals

NEUTRAL Classification - Document maintains balanced reporting through:
- Using neutral, descriptive language rather than partisan terminology
- Presenting multiple perspectives on political issues fairly
- Covering stories without clear preference for either party's framing
- Providing context that helps readers understand different viewpoints

Your Task:
Read the document below and determine which classification best fits based on the criteria above. Consider how the document's language, position, and coverage choices align with these patterns.

Document:
{content}

Classification:
1. LEFT
2. NEUTRAL
3. RIGHT

Respond with only the digit (1, 2, or 3) and no explanation.
ANSWER:""", {1: 'left', 2: 'neutral', 3: 'right'}),

    'zero_7': ("""
You are trained to identify political bias in news reporting and should classify documents into three categories: LEFT, NEUTRAL, or RIGHT political slant. Classify the political slant of this document using the following framework.

Classification Criteria:

LEFT Classification - The document shows bias toward left-leaning or progressive viewpoints through:
- Use of language associated with progressive or egalitarian values (e.g., "social justice," "climate emergency," "income inequality")
- Emphasizing themes such as redistribution, minority rights, environmentalism, or expanded public services
- Critiquing traditional power structures, free markets, or conservative cultural norms
- Framing stories in ways that support reformist or interventionist policies

RIGHT Classification - The document shows bias toward right-leaning or conservative viewpoints through:
- Use of language associated with conservative or traditional values (e.g., "personal responsibility," "illegal immigration," "family values")
- Emphasizing themes such as national sovereignty, law and order, market freedom, or traditional social structures
- Critiquing progressive movements, state intervention, or liberal cultural trends
- Framing stories in ways that support preservation of existing structures or national identity

NEUTRAL Classification - The document maintains balanced reporting through:
- Using neutral, descriptive language rather than ideologically charged terms
- Presenting multiple political or ideological perspectives fairly
- Covering issues with a focus on facts and evidence rather than advocacy
- Avoiding framing that favors one side over another

Your Task:
Read the document below and determine which classification best fits based on the criteria above. Consider how the document's language, framing, and topic emphasis align with these patterns.

Document:
{content}

Classification:
1. LEFT
2. NEUTRAL
3. RIGHT

Respond with only the digit (1, 2, or 3) and no explanation.
ANSWER:""", {1: 'left', 2: 'neutral', 3: 'right'}),

    'zero_8': ("""
You are an expert political analyst tasked with classifying documents by their political slant based on three key factors: language choices, political positions, and issue coverage. Please analyze the following document and classify its political slant as LEFT, NEUTRAL, or RIGHT.

Consider these three factors in your analysis:

Language Analysis: Look for partisan terminology
- Right-leaning terms: "death tax," "job creators," "government overreach," "pro-life"
- Left-leaning terms: "estate tax," "wealthy elite," "corporate welfare," "pro-choice"

Political Position: Assess alignment with party positions
- Right-leaning: supports conservative policies (lower taxes, deregulation, traditional values)
- Left-leaning: supports progressive policies (social programs, regulation, social justice)

Coverage Selection: Evaluate what aspects are emphasized or omitted
- Does the document present only one side of a multi-faceted issue?
- Are counterarguments or alternative perspectives acknowledged?

Document to analyze:
{content}

Classification: Based on your analysis, classify this document as:
1. LEFT: Shows clear bias toward progressive viewpoints
2. NEUTRAL: Presents balanced coverage without clear partisan lean
3. RIGHT: Shows clear bias toward conservative viewpoints

Respond with only the digit (1, 2, or 3) and no explanation.
ANSWER:""", {1: 'left', 2: 'neutral', 3: 'right'}),

    'Zero-shot 5': ("""
Given the current political climate, you are a fact-checker for a highly reputable news organization. Your task is to analyze the political slant of various documents based on their content and context. Your analysis will play a crucial role in maintaining the organization's integrity and credibility.

Classify the political slant of the document as either:
1. Left-leaning
2. Neutral
3. Right-leaning

Document:
{content}

Classification:
1. LEFT
2. NEUTRAL
3. RIGHT

Respond with only the digit (1, 2, or 3) and no explanation.
ANSWER:""", {1: 'left', 2: 'neutral', 3: 'right'}),

#     'Zero-shot 2': ("""
# You are an expert in political bias. Classify the document below as LEFT (1), NEUTRAL (2), or RIGHT (3) based on:

# - Language (partisan terms)
# - Position (alignment with progressive or conservative policies)
# - Framing (balance vs. one-sidedness)

# Use only the document content. Ignore source or author.

# DOCUMENT:
# {content}

# Respond with only the digit (1, 2, or 3) and no explanation.
# ANSWER:""", {1: 'left', 2: 'neutral', 3: 'right'}),

#     'Zero-shot 4': ("""
# You are an expert in political language. Classify the document below as LEFT (1), NEUTRAL (2), or RIGHT (3) based on:

# - Language (partisan terms)
# - Position (alignment with progressive or conservative policies)
# - Framing (balance vs. one-sidedness)

# Use only the content of the document. Ignore source or author.

# DOCUMENT:
# {content}

# Classification:
# 1. LEFT
# 2. NEUTRAL
# 3. RIGHT

# Respond with only the digit (1, 2, or 3) and no explanation.
# ANSWER:""", {1: 'left', 2: 'neutral', 3: 'right'}),

    'Zero-shot-3v2': ("""
Analyze the following document to determine its political orientation. Classify it as LEFT, NEUTRAL, or RIGHT based on the following three factors:

A. Language Bias - Detect politically charged wording.
- Right-leaning: "death tax," "job creators," "government overreach," "pro-life"
- Left-leaning: "estate tax," "corporate welfare," "wealthy elite," "pro-choice"

B. Ideological Positioning - Identify alignment with political ideologies.
- Right: favors conservative ideas (free market, strong borders, traditional values)
- Left: favors progressive ideas (social equity, climate action, government programs)

C. Framing and Coverage - Assess the narrative balance.
- Is only one side presented?
- Are alternate viewpoints dismissed or included?

Focus on the document's content alone. Ignore tone or assumed author intent.

DOCUMENT:
{content}

Classification, choose ONE:
1. LEFT (clearly favors progressive/Democratic positions)
2. NEUTRAL (balanced presentation)
3. RIGHT (clearly favors conservative/Republican positions)

Respond with only the digit (1, 2, or 3) and no explanation.
ANSWER:""", {1: 'left', 2: 'neutral', 3: 'right'}),

#     'Zero-shot 3': ("""
# Analyze the following document to determine its political orientation. Classify it as LEFT, NEUTRAL, or RIGHT based on the following three factors:

# A. Language Bias - Detect politically charged wording.
# - Right-leaning: "death tax," "job creators," "government overreach," "pro-life"
# - Left-leaning: "estate tax," "corporate welfare," "wealthy elite," "pro-choice"

# B. Ideological Positioning - Identify alignment with political ideologies.
# - Right: favors conservative ideas (free market, strong borders, traditional values)
# - Left: favors progressive ideas (social equity, climate action, government programs)

# C. Framing and Coverage - Assess the narrative balance.
# - Is only one side presented?
# - Are alternate viewpoints dismissed or included?

# Focus on the document's content alone. Ignore tone or assumed author intent.

# DOCUMENT:
# {content}

# Classification, choose ONE:
# 1. LEFT (clearly favors progressive/Democratic positions)
# 2. NEUTRAL (balanced presentation)
# 3. RIGHT (clearly favors conservative/Republican positions)

# Respond with only the digit (1, 2, or 3) and no explanation.
# ANSWER:""", {1: 'left', 2: 'neutral', 3: 'right'}),

    'zero_4_v4': ("""
You are a political media analyst. Classify the following document as having a political slant of LEFT, NEUTRAL, or RIGHT. Use ONLY the criteria below:

A. Language Use:
- Right-leaning: phrases like "death tax", "job creators", "government overreach", "pro-life"
- Left-leaning: phrases like "estate tax", "wealthy elite", "corporate welfare", "pro-choice"

B. Political Position:
- Right: supports conservative values (e.g., deregulation, tax cuts, national identity)
- Left: supports progressive values (e.g., social justice, environmental protection, welfare expansion)

C. Issue Framing:
- Bias exists if the document selectively emphasizes or omits sides
- NEUTRAL documents present multiple views or acknowledge counterarguments

Evaluate only the document's content. Do not speculate about the source or intent.

DOCUMENT:
{content}

Classification:
1. LEFT
2. NEUTRAL
3. RIGHT

Respond with only the digit (1, 2, or 3) and no explanation.
ANSWER:""", {1: 'left', 2: 'neutral', 3: 'right'}),

    }