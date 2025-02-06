sbatch \
    -n 16 \
    --mem-per-cpu=1000 \
    --gpus=rtx_4090:1 \
    --time=1440 \
    --wrap="\
    bash /cluster/home/anqiali/MasterThesis/scripts/train_transformer.bash \
    batch_size=2048 \
    epochs=1000 \
    action_window_size=20 \
    window_size=50 \
    noise_enhance_coef=0.01 \
    model.vqvae_model.vqvae_groups=2 \
    model.vqvae_model.vqvae_n_embed=16 \
    model.vqvae_model.n_latent_dims=512 \
    model.res_iter=true \
    data.data_directory=/cluster/scratch/anqiali/data_pt/Task_MM/No_Randomization/state_dataset.h5 \
    vqvae_load_dir=/cluster/home/anqiali/MasterThesis/checkpoints/oven_vq_vae/2025-02-04/10-01-28/pretrain_oven/MM_NoRandomization/trained_vqvae.pt \
    wandb.run_name=transformer/MM_NoRandomization_gaussian"

