# Batch-confounding diagnostic

## Evaluated population and available comparisons

All input rows are included: **57 samples**, 40 positive and 17 negative.
**2 mixed batches** contain both labels, providing 116 within-batch positive-negative pairs.
Mixed batches contain 37 samples (29 positive; 8 negative). There are 3 pure batches.
Mixed batch IDs: &quot;2&quot;, &quot;5&quot;.
Pure batch IDs: &quot;3&quot;, &quot;1&quot;, &quot;4&quot;.
These are observed comparisons, not established independent or causal contrasts.

## Exact design rank

Design: one indicator per observed batch, without an additional intercept.
Batch rank: **5**; batch-plus-target rank: **6**.
The target adds a dimension. This does not establish an unconfounded or well-supported effect.

## Descriptive batch-prevalence score

**Score(i) = observed positives in sample i's batch / samples in that batch.**
The observed labels define and evaluate this in-sample score. It is not held-out predictive performance.
Uses observed labels: Yes.
AUC: **283/340** (decimal: 0.8323529411764706; status: Defined).
AUC = (wins + ties / 2) / positive-negative pairs.
Wins: 508; ties: 116; losses: 56; pairs: 680.

## Batch composition

| Batch | N | Positive | Negative | Prevalence | Contrast available | Pairs |
|---|---:|---:|---:|---|---|---:|
| &quot;3&quot; | 4 | 0 | 4 | 0/1 | No | 0 |
| &quot;2&quot; | 18 | 14 | 4 | 7/9 | Yes | 56 |
| &quot;5&quot; | 19 | 15 | 4 | 15/19 | Yes | 60 |
| &quot;1&quot; | 11 | 11 | 0 | 1/1 | No | 0 |
| &quot;4&quot; | 5 | 0 | 5 | 0/1 | No | 0 |

## Recorded outcomes by batch

These are descriptive metadata, not additional validated targets. Quoted strings are literal values; Missing (null) is a missing value.

| Batch | Outcome | N | Positive | Negative |
|---|---|---:|---:|---:|
| &quot;3&quot; | &quot;Normal&quot; | 4 | 0 | 4 |
| &quot;2&quot; | &quot;Normal&quot; | 4 | 0 | 4 |
| &quot;2&quot; | &quot;sTCC&#45;CIS&quot; | 13 | 13 | 0 |
| &quot;2&quot; | &quot;mTCC&quot; | 1 | 1 | 0 |
| &quot;5&quot; | &quot;sTCC&#43;CIS&quot; | 12 | 12 | 0 |
| &quot;5&quot; | &quot;sTCC&#45;CIS&quot; | 3 | 3 | 0 |
| &quot;5&quot; | &quot;Biopsy&quot; | 4 | 0 | 4 |
| &quot;1&quot; | &quot;mTCC&quot; | 11 | 11 | 0 |
| &quot;4&quot; | &quot;Biopsy&quot; | 5 | 0 | 5 |

## Subject metadata

Status: Absent; unique subjects: Not available.
Subject independence is not verified. Missing subject metadata cannot establish independent samples.

## Schedule invariance

Fixed-score AUC remained unchanged for all supplied draws. This is expected and is not a scientific-validity pass.
Draws checked: 100.
RNG replay: Verified.
Schedule internal integrity SHA-256: &quot;f7d5181200eaceb40692db9c97c0514e1aaba1e3e7991ece5d3470a0dbba5b97&quot;.
Within-batch shuffles preserve batch class counts and therefore this score's AUC, when defined. Expected invariance does not show that leakage is absent. Pure batches cannot change label assignments.

## Input identity

| Binding field | Value |
|---|---|
| &quot;shape&quot; | &quot;&#91;57, 22283&#93;&quot; |
| &quot;matrix&#95;convention&quot; | &quot;&#60;f8 C&quot; |
| &quot;matrix&#95;sha256&quot; | &quot;0c9945e6acc19d8fc24d29d8e91b920dc8803def02c115bdecd2a380fdec3581&quot; |
| &quot;ordered&#95;metadata&#95;sha256&quot; | &quot;2e50ab7dd8a1cd0f53836f86463a71424e9ba5b1ef82db0a2c93fbfcab24c208&quot; |
| &quot;dataset&#95;sha256&quot; | &quot;0d6b58d3952f68a12667e8c0eb06cb9ce92e1f8891021044d5546c9413c4afcf&quot; |

Diagnostic input SHA-256: &quot;a65187d0eaca0cbb9595fdef355fa886beca66a337053a5af07668ec79c5bb10&quot;.
INPUTS.json separately records the exact input-file snapshot hashes. A schedule file hash and its internal integrity digest identify different representations.

## Limitations

This descriptive report does not select or fit a model, provide p-values, establish exchangeability, resolve biological confounding, or validate clinical use. AUC 0.5 does not rule out other shortcuts. Downstream schedule evaluation must refit the complete label-dependent preprocessing, selection and model pipeline.

&quot;The batch&#45;prevalence score uses observed labels and is in&#45;sample descriptive evidence, not held&#45;out predictive performance&#46;&quot;

&quot;Available within&#45;batch contrasts and design rank do not establish causal identification, stable estimation, or generalization&#46;&quot;

&quot;Within&#45;batch randomization preserves batch label counts and cannot test discrimination arising solely from batch composition&#46;&quot;

&quot;Within&#45;batch exchangeability and biological subject independence are unverified scientific assumptions&#46;&quot;

&quot;Recorded subject IDs are metadata; absent or unique IDs do not establish independence&#46;&quot;

&quot;Any downstream label&#45;dependent pipeline requires complete refitting for each draw, including selection, tuning, fitting and evaluation&#46;&quot;

&quot;No p&#45;values, fitted models, causal conclusions, or production&#45;engine integration are supplied&#46;&quot;
