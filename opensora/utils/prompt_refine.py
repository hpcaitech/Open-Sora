import base64
import os
import re
from mimetypes import guess_type

from openai import OpenAI

MINIMAX_BASE_URL = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1")

MINIMAX_MODELS = [
    "MiniMax-M2.7",
    "MiniMax-M2.7-highspeed",
]


def _get_client(model: str) -> OpenAI:
    """Return an OpenAI-compatible client configured for the given model."""
    if model.startswith("MiniMax"):
        api_key = os.environ.get("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError(
                "MINIMAX_API_KEY environment variable is not set. "
                "Please set it to use MiniMax models."
            )
        return OpenAI(api_key=api_key, base_url=MINIMAX_BASE_URL)
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def has_minimax_key() -> bool:
    """Check whether the MiniMax API key is configured."""
    return bool(os.environ.get("MINIMAX_API_KEY"))


def _strip_think_tags(text: str) -> str:
    """Remove chain-of-thought <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extra_tokens(model: str) -> int:
    """Extra token budget for MiniMax chain-of-thought <think> blocks."""
    return 500 if model.startswith("MiniMax") else 0

sys_prompt_t2v = """You are part of a team of bots that creates videos. The workflow is that you first create a caption of the video, and then the assistant bot will generate the video based on the caption. You work with an assistant bot that will draw anything you say.

For example, outputting "a beautiful morning in the woods with the sun peaking through the trees" will trigger your partner bot to output an video of a forest morning, as described. You will be prompted by people looking to create detailed, amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.

There are a few rules to follow:

You will only ever output a single video description per user request.

You should not simply make the description longer.

Video descriptions must have the same num of words as examples below. Extra words will be ignored.
"""

sys_prompt_t2i = """You are part of a team of bots that creates videos. The workflow is that you first create an image caption for the first frame of the video, and then the assistant bot will generate the video based on the image caption.

For example, outputting "a beautiful morning in the woods with the sun peaking through the trees" will trigger your partner bot to output an image of a forest morning, as described. You will be prompted by people looking to create detailed, amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.

There are a few rules to follow:

You will only ever output a single image description per user request.

You should not simply make the description longer.

Image captions must have the same num of words as examples. Extra words will be ignored.

Note: The input image is the first frame of the video, and the output image caption should include dynamic information.

Note: Don't contain camera transitions!!! Don't contain screen switching!!! Don't contain perspective shifts !!!

Note: Use daily language to describe the video, don't use complex words or phrases!!!
"""

sys_prompt_i2v = """You are part of a team of bots that creates videos. The workflow is that you first create a caption of the video based on the image, and then the assistant bot will generate the video based on the caption. You work with an assistant bot that will draw anything you say.

Give a highly descriptive video caption based on input image and user input. As an expert, delve deep into the image with a discerning eye, leveraging rich creativity, meticulous thought. When describing the details of an video, include appropriate dynamic information to ensure that the video caption contains reasonable actions and plots. If user input is not empty, then the caption should be expanded according to the user's input.

The input image is the first frame of the video, and the output video caption should describe the motion starting from the current image. User input is optional and can be empty.

Answers should be comprehensive, conversational, and use complete sentences. The answer should be in English no matter what the user's input is. Provide context where necessary and maintain a certain tone.  Begin directly without introductory phrases like "The image/video showcases" "The photo captures" and more. For example, say "A scene of a woman on a beach", instead of "A woman is depicted in the image".

Note: Must include appropriate dynamic information like actions, plots, etc. If the user prompt did not contain any dynamic information, then you must add some proper dynamic information like actions to make the video move!!!

Note: Try begin the sentence with phrases like  "A scene of" or "A view of" or "A close-up of" to make the video more descriptive!!!

Note: Use daily language to describe the video, don't use complex words or phrases!!!
"""

sys_prompt_motion_score = """
We define a video’s motion score as its FFMPEG VMAF motion value. We now have a video generation model that accepts a desired VMAF motion value as input. To reduce user burden, please predict an optimal motion score for generating a high-quality video based on the user’s text prompt. For reference:
	•	For runway videos featuring models, a motion score of 4 is ideal.
	•	For static videos, a motion score of 1 is preferred.

Output format:
“{} motion score”, where {} is an integer between 1 and 15.

User input:
"""


def image_to_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{base64_encoded_data}"


def refine_prompt(
    prompt: str,
    retry_times: int = 3,
    type: str = "t2v",
    image_path: str = None,
    model: str = None,
):
    """
    Refine a prompt to a format that can be used by the model for inference.

    Args:
        prompt: The input prompt text.
        retry_times: Number of retry attempts on failure.
        type: Prompt type - "t2v" (text-to-video), "t2i" (text-to-image),
              "i2v" (image-to-video), or "motion_score".
        image_path: Path to reference image (required when type="i2v").
        model: LLM model to use for refinement. Defaults to the PROMPT_MODEL
               environment variable, or "gpt-4o" if unset. Use MiniMax models
               (e.g. "MiniMax-M2.7") together with MINIMAX_API_KEY.
    """
    if model is None:
        model = os.environ.get("PROMPT_MODEL", "gpt-4o")

    client = _get_client(model)
    extra = _extra_tokens(model)

    text = prompt.strip()
    response = None
    for i in range(retry_times):
        if type == "t2v":
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"{sys_prompt_t2v}"},
                    {
                        "role": "user",
                        "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "A street with parked cars on both sides, lined with commercial buildings featuring Korean signs. The overcast sky suggests early morning or late afternoon."',
                    },
                    {
                        "role": "assistant",
                        "content": "A view of a street lined with parked cars on both sides. the buildings flanking the street have various signs and advertisements, some of which are in korean, indicating that this might be a location in south korea. the sky is overcast, suggesting either early morning or late afternoon light. the architecture of the buildings is typical of urban commercial areas, with storefronts on the ground level and possibly offices or residences above.",
                    },
                    {
                        "role": "user",
                        "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "Hands with rings and bracelets wash small greenish-brown seeds in a blue basin under running water, likely outdoors."',
                    },
                    {
                        "role": "assistant",
                        "content": "A close-up shot of a person's hands, adorned with rings and bracelets, washing a pile of small, round, greenish-brown seeds in a blue plastic basin. the water is running from an unseen source, likely a tap, and the person is using their hands to agitate the seeds, presumably to clean them. the background is indistinct but appears to be an outdoor setting with natural light.",
                    },
                    {
                        "role": "user",
                        "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "Three men stand near an open black car in a parking lot, with parked vehicles and a partly cloudy sky in the background."',
                    },
                    {
                        "role": "assistant",
                        "content": "A scene showing three men in an outdoor setting, likely a parking lot. the man on the left is wearing a light blue shirt and dark shorts, the man in the middle is dressed in a white shirt with a pattern and dark shorts, and the man on the right is wearing a green shirt and jeans. they are standing near a black car with its door open. in the background, there are parked vehicles, including a white truck and a red trailer. the sky is partly cloudy, suggesting it might be a sunny day.",
                    },
                    {
                        "role": "user",
                        "content": f'Create an imaginative video descriptive caption or modify an earlier caption in ENGLISH for the user input: " {text} "',
                    },
                ],
                model=model,  # glm-4-plus, gpt-4o, and MiniMax-M2.7 have been tested
                temperature=0.01,
                top_p=0.7,
                stream=False,
                max_tokens=250 + extra,
            )
        elif type == "t2i":
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"{sys_prompt_t2i}"},
                    {
                        "role": "user",
                        "content": 'Create an imaginative image descriptive caption or modify an earlier caption for the user input : "a girl on the beach"',
                    },
                    {
                        "role": "assistant",
                        "content": "A radiant woman stands on a deserted beach, arms outstretched, wearing a beige trench coat, white blouse, light blue jeans, and chic boots, against a backdrop of soft sky and sea.",
                    },
                    {
                        "role": "user",
                        "content": 'Create an imaginative image descriptive caption or modify an earlier caption for the user input : "A man in a blue shirt"',
                    },
                    {
                        "role": "assistant",
                        "content": "A determined man in athletic attire, including a blue long-sleeve shirt, black shorts, and blue socks, against a backdrop of a snowy field.",
                    },
                    {
                        "role": "user",
                        "content": f'Create an imaginative image descriptive caption or modify an earlier caption in ENGLISH for the user input: " {text} "',
                    },
                ],
                model=model,  # glm-4-plus, gpt-4o, and MiniMax-M2.7 have been tested
                temperature=0.01,
                top_p=0.7,
                stream=False,
                max_tokens=250 + extra,
            )
        elif type == "i2v":
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"{sys_prompt_i2v}"},
                    {
                        "role": "user",
                        "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "A street with parked cars on both sides, lined with commercial buildings featuring Korean signs. The overcast sky suggests early morning or late afternoon."',
                    },
                    {
                        "role": "assistant",
                        "content": "A view of a street lined with parked cars on both sides. the buildings flanking the street have various signs and advertisements, some of which are in korean, indicating that this might be a location in south korea. the sky is overcast, suggesting either early morning or late afternoon light. the architecture of the buildings is typical of urban commercial areas, with storefronts on the ground level and possibly offices or residences above.",
                    },
                    {
                        "role": "user",
                        "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "Hands with rings and bracelets wash small greenish-brown seeds in a blue basin under running water, likely outdoors."',
                    },
                    {
                        "role": "assistant",
                        "content": "A close-up shot of a person's hands, adorned with rings and bracelets, washing a pile of small, round, greenish-brown seeds in a blue plastic basin. the water is running from an unseen source, likely a tap, and the person is using their hands to agitate the seeds, presumably to clean them. the background is indistinct but appears to be an outdoor setting with natural light.",
                    },
                    {
                        "role": "user",
                        "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "Three men stand near an open black car in a parking lot, with parked vehicles and a partly cloudy sky in the background."',
                    },
                    {
                        "role": "assistant",
                        "content": "A scene showing three men in an outdoor setting, likely a parking lot. the man on the left is wearing a light blue shirt and dark shorts, the man in the middle is dressed in a white shirt with a pattern and dark shorts, and the man on the right is wearing a green shirt and jeans. they are standing near a black car with its door open. in the background, there are parked vehicles, including a white truck and a red trailer. the sky is partly cloudy, suggesting it might be a sunny day.",
                    },
                    {
                        "role": "user",
                        "content": f'Create an imaginative video descriptive caption or modify an earlier caption in ENGLISH for the user input based on the image: " {text} "',
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_to_url(image_path),
                                },
                            },
                        ],
                    },
                ],
                temperature=0.01,
                top_p=0.7,
                stream=False,
                max_tokens=250 + extra,
            )
        elif type == "motion_score":
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"{sys_prompt_motion_score}"},
                    {
                        "role": "user",
                        "content": f"{text}",
                    },
                ],
                model=model,  # glm-4-plus, gpt-4o, and MiniMax-M2.7 have been tested
                temperature=0.01,
                top_p=0.7,
                stream=False,
                max_tokens=100 + extra,
            )
        if response is None:
            continue
        if response.choices:
            return _strip_think_tags(response.choices[0].message.content)
    return prompt


def refine_prompts(
    prompts: list[str],
    retry_times: int = 3,
    type: str = "t2v",
    image_paths: list[str] = None,
    model: str = None,
):
    if image_paths is None:
        image_paths = [None] * len(prompts)
    refined_prompts = []
    for prompt, image_path in zip(prompts, image_paths):
        refined_prompt = refine_prompt(
            prompt, retry_times=retry_times, type=type, image_path=image_path, model=model
        )
        refined_prompts.append(refined_prompt)
    return refined_prompts


def refine_prompts_by_minimax(
    prompts: list[str],
    retry_times: int = 3,
    type: str = "t2v",
    image_paths: list[str] = None,
    model: str = "MiniMax-M2.7",
):
    """Refine prompts using the MiniMax API (requires MINIMAX_API_KEY)."""
    return refine_prompts(prompts, retry_times=retry_times, type=type, image_paths=image_paths, model=model)
