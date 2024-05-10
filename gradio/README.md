# 🕹 Gradio Demo

We have provided a Gradio demo app for you to generate videos via a web interface. You can choose to run it locally or deploy it to Hugging Face by following the instructions given below.

## 🚀 Run Gradio Locally

We assume that you have already installed `opensora` based on the instructions given in the [main README](../README.md). Follow the steps below to run this app on your local machine.

1. First of all, you need to install `gradio` and `spaces`.

```bash
pip install gradio spaces
```

2. Afterwards, you can use the following command to launch different models. Remember to launch the command in the project root directory instead of the `gradio` folder.

```bash
# run the default model v1-HQ-16x256x256
python gradio/app.py

# run the model with higher resolution
python gradio/app.py --model-type v1-HQ-16x512x512

# run with a different host and port
python gradio/app.py --port 8000 --host 0.0.0.0

# run with acceleration such as flash attention and fused norm
python gradio/app.py --enable-optimization

# run with a sharable Gradio link
python gradio/app.py --share
```

3. You should then be able to access this demo via the link which appears in your terminal.


## 📦 Deploy Gradio to Hugging Face Space

We have also tested this Gradio app on Hugging Face Spaces. You can follow the steps below.

1. Create a Space on Hugging Face, remember to choose `Gradio SDK` and GPU space hardware.

2. Clone the Space repository in your local machine.

3. Copy the `configs` folder and `gradio/app.py` and `gradio/requirements.txt` to the repository you just cloned. The file structure will look like:

```text
- configs
    - opensora
        - inference
            - 16x256x256.py
            - 16x512x512.py
            - 64x512x512.py
        ...
    ...
- app.py
- requirements.txt
- README.md
- LICENSE
- ...
```

4. Push the files to your remote Hugging Face Spaces repository. The application will be built and run automatically.
