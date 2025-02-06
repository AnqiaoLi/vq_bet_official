sbatch \
    -n 16 \
    --mem-per-cpu=1000 \
    --gpus=rtx_4090:1 \
    --time=1440 \
    --wrap="\
    bash /cluster/home/anqiali/MasterThesis/scripts/train_vq_vae.bash \
    batch_size=1024 \
    action_window_size=5 \
    res_action=false \
    data.data_directory="/cluster/scratch/anqiali/data_pt/Task_MM/No_Randomization/state_dataset.h5" \
    vqvae_model.vqvae_groups=2 \
    vqvae_model.vqvae_n_embed=16 \
    vqvae_model.n_latent_dims=512 \
    vqvae_model.encoder_loss_multiplier=1 \
    normalizer.mode="limits" \
    save_path="/cluster/home/anqiali/MasterThesis/checkpoints/oven_vq_vae/$(date +%Y-%m-%d)/$(date +%H-%M-%S)" \
    wandb.tags=["no_randomization"] \
    wandb.run_name=pretrain_oven/MM_NoRandomization_res_limits_b1024"
