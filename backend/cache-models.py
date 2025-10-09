import os
import torch
from diffusers import AutoPipelineForImage2Image, AutoencoderTiny
from diffusion_processor import DiffusionProcessor


class DiffusionProcessor:
    def __init__(self, warmup=None, local_files_only=False):
        from diffusers import UNet2DConditionModel, EulerDiscreteScheduler
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        
        base_sdxl = "stabilityai/stable-diffusion-xl-base-1.0"
        lightning_model = "ByteDance/SDXL-Lightning"
        vae_model = "madebyollin/taesdxl"

        # Load base SDXL
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            base_sdxl,
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=local_files_only,
        )
        
        # Download and load Lightning UNet weights (8-step version)
        print("Loading SDXL-Lightning 8-step UNet...")
        unet_path = hf_hub_download(
            repo_id=lightning_model,
            filename="sdxl_lightning_8step_unet.safetensors",
            local_files_only=local_files_only,
        )
        unet_state_dict = load_file(unet_path)
        self.pipe.unet.load_state_dict(unet_state_dict)
        print("Lightning UNet loaded")
        
        # Use Lightning's scheduler
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config,
            timestep_spacing="trailing"
        )

        # Use Tiny VAE
        self.pipe.vae = AutoencoderTiny.from_pretrained(
            vae_model, torch_dtype=torch.float16, local_files_only=local_files_only
        )

        print("Models loaded")

    def save_models(
        self,
        save_directory_base,
        save_directory_vae,
        safe_serialization=True,
        variant=None,
        push_to_hub=False,
        **kwargs,
    ):
        # Ensure the save directories exist
        if not os.path.exists(save_directory_base):
            os.makedirs(save_directory_base)
        if not os.path.exists(save_directory_vae):
            os.makedirs(save_directory_vae)

        # Save the base model
        self.pipe.save_pretrained(
            save_directory_base,
            safe_serialization=safe_serialization,
            variant=variant,
            push_to_hub=push_to_hub,
            **kwargs,
        )
        print(f"Base model saved to {save_directory_base}")

        # Save the VAE model
        self.pipe.vae.save_pretrained(
            save_directory_vae,
            safe_serialization=safe_serialization,
            variant=variant,
            push_to_hub=push_to_hub,
            **kwargs,
        )
        print(f"VAE model saved to {save_directory_vae}")


# Example usage
processor = DiffusionProcessor(local_files_only=False)
save_dir_base = "./saved_pipeline/sdxl-lightning"
save_dir_vae = "./saved_pipeline/taesdxl"
processor.save_models(
    save_directory_base=save_dir_base,
    save_directory_vae=save_dir_vae,
    safe_serialization=True,
    variant="fp16",
    push_to_hub=False,  # Change to True if you want to push to Hugging Face hub
)
