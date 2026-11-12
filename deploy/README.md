# Deployment

## Hugging Face Spaces (live demo)

The app runs on a free [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu) Space, which borrows a GPU only
while a request is being answered. The Space is built from this repository by
[`huggingface/build_space.py`](huggingface/build_space.py) and published by the
[`deploy-space`](../.github/workflows/deploy-space.yml) workflow on every push to `main`.

One-time setup:

1. On Hugging Face, accept the [Gemma license](https://huggingface.co/google/gemma-2-2b-it) and create a
   **Gradio** Space with **ZeroGPU** hardware (for example `albertogomezsoteres/probity`).
2. In the Space settings, add a secret `HF_TOKEN` with a *read* token that has access to Gemma.
3. In this GitHub repository (*Settings > Secrets and variables > Actions*), add the secret `HF_TOKEN` with a
   *write* token for the Space, and the variable `HF_SPACE_ID` with the Space id.
4. Run the workflow once from the *Actions* tab (or push to `main`).

To preview the bundle locally: `python deploy/huggingface/build_space.py build/space`.

The demo cases in the chat tab are the cases found during the study. To verify them live on the deployed model,
run `probity app --find-demos` once on a GPU and commit the two `results/demo_cases*.json` files it writes.

## Docker

The [`Dockerfile`](../Dockerfile) installs the locked dependencies (`uv.lock`) and runs the `probity` CLI:

```bash
docker build -t probity .
docker run --gpus all -e HF_TOKEN -p 7860:7860 probity                          # the app
docker run --gpus all -e HF_TOKEN -v "$PWD/outputs:/app/outputs" probity run    # the full pipeline
```
