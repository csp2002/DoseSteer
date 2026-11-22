# DoseDiT

DoseDiT-S/4 (91.3 M parameters) keeps DoseDiff's inputs, diffusion objective, sampler and evaluation and replaces its UNet
with two ViT condition encoders (CT; the 20 mask and distance-map channels) and a DiT denoiser with U-ViT skips. The
encoders use DINOv3's ViT code (`third_party/dinov3`).
