# multimodal-rs-change-detection

"Multimodal framework for event-based change detection in remote sensing, fusing spatial-temporal imagery with semantic text using VLMs for interpretable analysis (e.g., urban expansion, flooding). Includes benchmarks on LEVIR-CC and RSICCformer."

## Gemini few-shot change detection

The `multimodal_rs_change_detection.gemini_few_shot` module implements a few-shot prompting workflow for Google Gemini's multimodal models. It consumes LEVIR-CC style `A` (pre-event) and `B` (post-event) folders, applies curated support examples, and stores the predictions—including captions, confidences, rationales, and category-aware change/no-change labels—in `results/few_shot_gemini/gemini-fewshots-result1.json` by default.

### Prerequisites

1. Create a `.env` file in the project root (or pass `--env-path`) with the variable `GOOGLE_API_KEY=...`. The loader will also fall back to `.env` files placed alongside `multimodal_rs_change_detection/gemini_few_shot.py`.
2. (Optional) Add `LEVIR_CC_DATASET=/path/to/Levir-CC/images/test` to the same `.env` file so the CLI and VS Code integrations can infer the dataset path. You can also set `LEVIR_CC_SUPPORT_DATASET` if your support examples live in a different folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure your dataset directory contains matching `A/*.png` and `B/*.png` image pairs. The
   provided few-shot support set expects the example filenames listed in
   `multimodal_rs_change_detection.__init__` to exist under these folders. If you maintain a
   separate directory for the support examples, pass it via `--support-dataset-path`.

### Running

```bash
python -m multimodal_rs_change_detection.gemini_few_shot \
  /path/to/Levir-CC-dataset/images/test \
  --num-samples 15 \
  --output-dir results/few_shot_gemini \
  --output-name gemini-fewshots-result1.json \
  --support-dataset-path /path/to/Levir-CC-dataset/images/test
```

If you set `LEVIR_CC_DATASET` in your environment or `.env` file, you can omit the positional dataset argument and rely on that value instead. Set `LEVIR_CC_SUPPORT_DATASET` to reuse the support examples from another dataset directory.

The script prints an aggregate summary and writes the full JSON payload. The JSON report records how many pairs were classified as `change` vs. `no_change`, with non-repeating captions derived from the Gemini response and the provided support examples.

### Running in VS Code

You can run and debug the workflow directly from VS Code using the provided launch configuration:

1. Install the Python extension and select your interpreter/virtual environment.
2. Add the following variables to your `.env` file (loaded automatically by VS Code and the script):
   ```text
   GOOGLE_API_KEY=your_api_key_here
   LEVIR_CC_DATASET=G:\\Change_Detection\\working octobar\\CD-work\\Levir-CC-dataset\\images\\test
   ```
   Adjust `LEVIR_CC_DATASET` to point at your local LEVIR-CC test directory.
3. The repo now includes `tasks.json` entries for installing dependencies and running the detector with the `.env` file automatically loaded. You can invoke them from the **Terminal → Run Task…** menu if you prefer a task runner over the debugger.
4. Open the **Run and Debug** panel and choose **Gemini Few-Shot Detection**. Press **F5** (or click the play button). The script will look for `LEVIR_CC_DATASET` if no positional dataset argument is provided and write the output JSON to `results/few_shot_gemini/gemini-fewshots-result1.json`.

Update the launch configuration or the task definition if you want to change arguments such as `--num-samples` or `--output-dir`.
