# NULLGARDEN

Executable negative controls for scientific machine learning. You load a labelled dataset with batches (or use the bundled bladderbatch-shaped designs), and NULLGARDEN reruns the same classifier under controls that remove the real signal but keep a suspected shortcut. If the score survives, the model was exploiting the shortcut, not learning the biology.

Built for bench scientists and ML reviewers who see a high cross-validated AUC on batch-structured data and need to know whether to believe it.

## Quick start

Requires Node 22 or newer. No npm dependencies.

```
export NEBIUS_API_KEY=...        # optional; enables the Nemotron features
node server.mjs                  # serves http://127.0.0.1:4412
node --test                      # 15 tests, none need the network
```

Open http://127.0.0.1:4412. Pick a dataset tab, press **Run controls**, read the verdict and the AUC ladder. With a key set, **Propose hypotheses** and **Explain results with Nemotron** are live. Without a key those two buttons show a plain message and everything else still works. `PORT` changes the port. A CSV can be loaded with the "Load CSV" button (columns: `sample_id,batch,label,feature...`; label 0 or 1; at least 10 samples).

## The controls

A fixed pipeline (top-20 feature selection by |t| inside each training fold, then diagonal LDA) is scored by cross-validation, then rerun under four controls: batch-only baseline, within-batch label permutation (with a shuffled-label null band), leave-one-batch-out, and batch-centered features. The rules in `src/core.mjs` turn the ladder of AUCs into a verdict. Nothing in that path calls a model.

## How it uses NVIDIA Nemotron on Nebius Token Factory

- **Endpoint and model:** `https://api.tokenfactory.nebius.com/v1/chat/completions` (OpenAI-compatible), model `nvidia/nemotron-3-super-120b-a12b` (override with `NULLGARDEN_MODEL`). The call is made server-side in `src/llm.mjs`; the key is read from `NEBIUS_API_KEY` and never sent to the browser.
- **Why this size:** the task is short structured reasoning over a small batch table. The 120B-A12B mixture-of-experts model follows the JSON format reliably and answers in about 3 to 10 seconds; a larger model would add latency without changing the checked output.
- **Propose hypotheses:** Nemotron reads the batch/label table and your note and names up to four shortcut hypotheses, each mapped to one control from the menu. The server drops any item whose control id is not on the menu. The app then runs the controls itself and marks each hypothesis supported or not supported from the computed AUCs. The model proposes; the controls decide.
- **Explain results:** Nemotron writes a short plain-language read-out of the computed table. Every decimal in its text is compared with the computed figures (tolerance 0.006) and the UI shows "N of M match", flagging any number the controls never produced.

## Honest limits

- The bundled datasets use the real bladderbatch batch/label counts (57 samples, 5 batches) but the expression values are synthetic.
- One fixed classifier, not your pipeline. The controls demonstrate the method; they are not a certification that a result is valid.
- Verdicts come from simple AUC thresholds on small samples and are heuristics.
- Nemotron output varies between calls; the figure check catches wrong numbers, not wrong reasoning.
- Experimental, not clinically validated.

## Repository components

The `components/` directory holds earlier standalone Python modules (batch-label schedules, batch-confounding diagnostic) contributed separately; see their READMEs. Provenance notes are in `docs/source-recovery.json`. The web app in `server.mjs`, `src/` and `public/` does not depend on them.

## License

MIT, see LICENSE.
