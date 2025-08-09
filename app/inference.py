import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

device = "cuda"
dtype = torch.float16
step = 4

# Load motion adapter
adapter = MotionAdapter().to(device, dtype)
adapter.load_state_dict(load_file("./app/models/animatediff/animatediff_lightning_4step_diffusers.safetensors", device=device))

# Load base model pipeline
pipe = AnimateDiffPipeline.from_pretrained(
    "./app/models/base",
    motion_adapter=adapter,
    torch_dtype=dtype
).to(device)

# Replace scheduler
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config,
    timestep_spacing="trailing",
    beta_schedule="linear"
)

def generate_video():
    prompt = "A girl smiling"
    output = pipe(prompt=prompt, guidance_scale=1.0, num_inference_steps=step)
    output.frames[0][0].save("output.gif", save_all=True, append_images=output.frames[0][1:], duration=100, loop=0)
    return "output.gif"
