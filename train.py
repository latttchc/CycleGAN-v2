import torch 
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator
from dataset import CustomDataset
from utils import save_checkpoint, load_checkpoint


def train_fn(disc_A, disc_B, gen_B, gen_A, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    A_reals = 0
    A_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (trainB, trainA) in enumerate(loop):
        trainB = trainB.to(config.DEVICE)
        trainA = trainA.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake_trainA = gen_A(trainB)
            D_A_real = disc_A(trainA)
            D_A_fake = disc_A(fake_trainA.detach())

            A_reals += D_A_real.mean().item()
            A_fakes += D_A_fake.mean().item()

            D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
            D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))
            D_A_loss = D_A_real_loss + D_A_fake_loss

            fake_trainB = gen_B(trainA)
            D_B_real = disc_B(trainB)
            D_B_fake = disc_B(fake_trainB.detach())

            D_B_real_loss = mse(D_B_real, torch.ones_like(D_B_real))
            D_B_fake_loss = mse(D_B_fake, torch.zeros_like(D_B_fake))
            D_B_loss = D_B_real_loss + D_B_fake_loss

            D_loss = (D_A_loss + D_B_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update

        with torch.cuda.amp.autocast():
            D_A_fake = disc_A(fake_trainA)
            D_B_fake = disc_B(fake_trainB)
            loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))
            loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake))

            cycle_trainB = gen_B(fake_trainA)
            cycle_trainA = gen_A(fake_trainB)
            cycle_trainA_loss = l1(trainB, cycle_trainB)
            cycle_trainB_loss = l1(trainA, cycle_trainA)

            identity_trainB = gen_B(trainB)
            identity_trainA = gen_A(trainA)
            identity_trainB_loss = l1(trainB, identity_trainB)
            identity_trainA_loss = l1(trainA, identity_trainA)

            G_loss = (
                loss_G_B +
                loss_G_A +
                cycle_trainB_loss * config.LAMBDA_CYCLE +
                cycle_trainA_loss * config.LAMBDA_CYCLE +
                identity_trainA_loss *config.LAMBDA_IDENTITY +
                identity_trainB_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_trainA * 0.5 + 0.5, f"saved_images/fake_trainA_{idx}.png")
            save_image(fake_trainB * 0.5 + 0.5, f"saved_images/fake_trainB_{idx}.png")

        loop.set_postfix(
            A_real=A_reals / (idx + 1),
            A_fake=A_fakes / (idx + 1)
        )


def main():
    disc_A = Discriminator(in_channels=3).to(config.DEVICE)
    disc_B = Discriminator(in_channels=3).to(config.DEVICE)
    gen_B = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_A = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )
    
    opt_gen = optim.Adam(
        list(gen_B.parameters()) + list(gen_A.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_CRITIC_A, disc_A, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_B, disc_B, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_A, gen_A, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_B, gen_B, opt_gen, config.LEARNING_RATE)

    dataset = CustomDataset(root_a=config.TRAIN_DIR + "/trainA", root_b=config.TRAIN_DIR + "/trainB", transform=config.transforms)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    d_scaler = torch.cuda.amp.GradScaler()
    g_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch [{epoch}/{config.NUM_EPOCHS}]")
        train_fn(disc_A, disc_B, gen_B, gen_A, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler)

        if config.SAVE_MODEL:
            save_checkpoint(gen_A, opt_gen, filename=config.CHECKPOINT_GEN_A)
            save_checkpoint(gen_B, opt_gen, filename=config.CHECKPOINT_GEN_B)
            save_checkpoint(disc_A, opt_disc, filename=config.CHECKPOINT_CRITIC_A)
            save_checkpoint(disc_B, opt_disc, filename=config.CHECKPOINT_CRITIC_B)

if __name__ == "__main__":
    main()