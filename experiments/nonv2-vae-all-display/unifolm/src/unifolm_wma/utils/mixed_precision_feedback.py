"""Keep original FP32 feedback weights while converting the video backbone."""
import torch


def cast_sampler_streams(video, action, state, precision, protect_feedback):
    if precision != 16:
        return video, action, state
    video = video.to(torch.float16)
    if not protect_feedback:
        action = action.to(torch.float16)
        state = state.to(torch.float16)
    return video, action, state


def convert_half_with_fp32_feedback(diffusion):
    # Never half() the whole module first: float() cannot recover its weights.
    protected = {"action_unet", "state_unet"}
    if any(True for _ in diffusion.parameters(recurse=False)):
        raise ValueError("Review direct denoiser parameters before mixed conversion")
    children = dict(diffusion.named_children())
    if not protected <= children.keys():
        raise ValueError("Missing action/state feedback heads")
    protected_ids = set()
    for name, child in children.items():
        if name in protected:
            if any(p.is_floating_point() and p.dtype != torch.float32
                   for p in child.parameters()):
                raise ValueError("Feedback weights must enter conversion as FP32")
            protected_ids.update(id(module) for module in child.modules())
        else:
            child.half()
    diffusion.dtype = torch.float16
    diffusion._fp32_feedback_heads = True
    return protected_ids
