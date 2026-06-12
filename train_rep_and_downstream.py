import copy
import sys
import time

import torch.optim as optim
import datetime
import matplotlib

matplotlib.use('Agg')
from train_utils import *
from image_utils import *
import pyrallis


def train():
    args = parse_args()

    full_config = pyrallis.encode(args)

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    wandb.init(project="imaging",
               name=f"run_{current_date}",
               config=full_config)
    run_id = wandb.run.id
    checkpoint_dir = os.path.join("checkpoints", run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    set_random_seed(args.random_seed)

    rep_model, downstream_model = create_rep_and_downstream_models(args, Path(checkpoint_dir), save_config=True)
    device = rep_model.device

    train_loader, test_loader = get_train_and_test_loader(args.dataset, args.data_loader)

    # Initialize best model tracking
    best_rep_loss = float('inf')
    best_downstream_loss = float('inf')
    best_rep_model_state = None
    best_downstream_model_state = None
    best_rep_model_for_best_downstream_state = None

    time_loggings = {}

    # Initialize the optimizer
    # optimizer both for representation learning and downstream model
    optimizer_name_to_class = {"Adam": optim.Adam, "SGD": optim.SGD, "AdamW": optim.AdamW}
    optimizer_class = optimizer_name_to_class.get(args.optimizer.name, None)
    if optimizer_class is None:
        raise ValueError(f"Invalid optimizer: {args.optimizer.name}")

    rep_model_params = list(rep_model.parameters())
    optimizer_params = rep_model_params if downstream_model is None else rep_model_params + list(
        downstream_model.parameters())
    optimizer = optimizer_class(optimizer_params, lr=args.optimizer.learning_rate)

    if args.optimizer.learning_rate_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0)
    elif args.optimizer.learning_rate_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.optimizer.lr_scheduler_step_size, gamma=0.1)
    elif args.optimizer.learning_rate_scheduler == "const":
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=args.num_epochs)
    else:
        raise ValueError(f"Invalid learning rate scheduler: {args.optimizer.learning_rate_scheduler}")

    # Calculate mean and std of data
    (train_mean_locations, train_std_locations,
     train_mean_forces, train_std_forces) = get_dataset_mean_std(args.data_processing, train_loader, device,
                                                                 args.constant_force_norm)
    img_coords = get_image_xy_locations(args.dataset)
    img_coords = img_coords.to(device)
    img_coords = (img_coords - train_mean_locations.mean(0)[:2]) / (train_std_locations.mean(0)[:2])

    print("Starting training...")
    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        rep_model.train()
        if downstream_model is not None:
            downstream_model.train()
        train_loss = 0
        train_rep_losses = {}
        train_rep_loss = 0
        train_downstream_loss = 0
        train_grads_max = 0
        train_grads_sum = 0
        train_grads_count = 0
        time_loggings["train_data_preparation"] = 0
        time_loggings["train_rep_model_forward"] = 0
        time_loggings["train_downstream_model_forward"] = 0
        time_loggings["train_loss_calculation"] = 0
        time_loggings["train_backward"] = 0
        train_start_time = time.time()
        for _, locations, forces, _, model_images, _, exp_name, traj_props, padding_mask in train_loader:
            start_data_preprocess = time.time()
            traj_lengths = traj_props[:, 5, :].long() if traj_props.shape[1] > 5 else None
            locations, forces, model_images = data_preprocess(
                locations, forces, model_images, train_mean_locations, train_std_locations,
                train_mean_forces, train_std_forces, args.dont_norm_locations, args.relative_locations,
                args.zero_location, args.zero_forces, args.shuffle_order, args.dataset.num_training_trajs,
                rep_model.device, traj_lengths=traj_lengths)
            padding_mask = padding_mask.to(rep_model.device)

            before_model_forward = time.time()
            time_loggings["train_data_preparation"] += before_model_forward - start_data_preprocess

            optimizer.zero_grad()
            representation, rep_loss, rep_loss_info, rep_inference_results = rep_model(forces, locations,
                                                                                        padding_mask=padding_mask)
            after_rep_model_forward = time.time()
            time_loggings["train_rep_model_forward"] += after_rep_model_forward - before_model_forward

            if args.models.downstream_model.name == "PhantomIndexClassifier":
                downstream_kwargs = {"exp_name": exp_name}
            elif args.models.downstream_model.name in ["TransposedConvParRepPred", "TransposedConvParRepPred3D"]:
                downstream_kwargs = {
                    "particles_locations": rep_inference_results["particles_locations"],
                    "active_particles_mask": rep_inference_results["active_particles_mask"],
                    "img_coords": img_coords
                }
            else:
                downstream_kwargs = {}

            random_representation = representation.unsqueeze(1)
            if "all_outputs" in rep_inference_results and (
                    args.optimizer.use_first_rep_for_downstream_training or args.optimizer.num_random_rep_for_downstream_training != -1):
                if args.optimizer.num_random_rep_for_downstream_training != -1:
                    # random sample num_random_rep_for_downstream_training from all_outputs, also support batch
                    random_indices = torch.randint(0, rep_inference_results["all_outputs"].shape[1], (
                        representation.shape[0], args.optimizer.num_random_rep_for_downstream_training)).to(
                        rep_model.device)
                else:
                    # use the last representation
                    random_indices = torch.full((representation.shape[0], 1),
                                                rep_inference_results["all_outputs"].shape[1] - 1,
                                                dtype=torch.long, device=rep_model.device)

                if args.optimizer.use_first_rep_for_downstream_training:
                    random_indices = torch.cat(
                        [torch.zeros((representation.shape[0], 1), dtype=torch.long,
                                     device=rep_model.device), random_indices], dim=1)

                random_indices = random_indices.unsqueeze(-1).expand(-1, -1,
                                                                     rep_inference_results["all_outputs"].size(
                                                                         -1))  # [B, K, D]

                random_representation = torch.gather(rep_inference_results["all_outputs"], dim=1, index=random_indices)

            if not args.dont_detach_representation:
                random_representation = random_representation.detach()

            tot_downstream_loss = torch.tensor(0.0, device=rep_model.device)
            if downstream_model is not None:
                for i in range(random_representation.shape[1]):
                    curr_representation = random_representation[:, i]
                    if args.zero_representation:
                        curr_representation = torch.zeros_like(curr_representation)
                    downstream_prediction, downstream_loss, downstream_loss_info, _ = downstream_model(
                        curr_representation,
                        model_images,
                        is_train=True,
                        **downstream_kwargs)
                    tot_downstream_loss += downstream_loss
                tot_downstream_loss = tot_downstream_loss / random_representation.shape[1]

            after_downstream_model_forward = time.time()
            time_loggings["train_downstream_model_forward"] += after_downstream_model_forward - after_rep_model_forward

            total_loss = args.optimizer.force_reconstruction_weight * rep_loss + args.optimizer.downstream_loss_weight * tot_downstream_loss

            # Compute loss
            train_loss += total_loss.item()
            for key, value in rep_loss_info.items():
                if key not in train_rep_losses:
                    train_rep_losses[key] = 0
                train_rep_losses[key] += value.item()

            train_downstream_loss += tot_downstream_loss.item()
            train_rep_loss += rep_loss.item()

            # Backward pass and optimize
            total_loss.backward()

            if args.optimizer.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(rep_model.parameters(), args.optimizer.gradient_clipping)
                if downstream_model is not None:
                    torch.nn.utils.clip_grad_norm_(downstream_model.parameters(), args.optimizer.gradient_clipping)

            grads = []
            relevent_models = []
            if not args.optimizer.freeze_rep_learning_model:
                relevent_models.append(rep_model)
            if downstream_model is not None:
                relevent_models.append(downstream_model)

            for model in relevent_models:
                for param in model.parameters():
                    if param.requires_grad:
                        grads.append(param.grad.view(-1))

            grads = torch.cat(grads) if len(grads) > 0 else torch.tensor([0.0])
            train_grads_max = max(train_grads_max, torch.max(torch.abs(grads)).item())
            train_grads_sum += torch.sum(torch.abs(grads)).item()
            train_grads_count += grads.numel()

            optimizer.step()
            after_backward = time.time()
            time_loggings["train_backward"] += after_backward - after_downstream_model_forward

        average_train_loss = train_loss / len(train_loader)
        average_train_rep_loss = train_rep_loss / len(train_loader)

        for key, value in train_rep_losses.items():
            train_rep_losses[key] = value / len(train_loader)

        average_train_downstream_loss = train_downstream_loss / len(train_loader)
        average_train_grads = train_grads_sum / train_grads_count
        time_loggings["train total time"] = time.time() - train_start_time

        rep_model.eval()  # Set model to evaluation mode
        if downstream_model is not None:
            downstream_model.eval()  # Set model to evaluation mode
            downstream_model.reset_metrics()
        test_loss = 0
        test_downstream_loss = 0
        test_rep_loss = 0
        test_rep_losses = {}

        eval_start_time = time.time()
        rep_inference_results = {}
        vis_rep_inference_results = {}
        # draw random batch index for visualization
        vis_batch_index = torch.randint(0, len(test_loader), (1,)).item()
        vis_forces = None
        vis_downstream_results = None
        vis_model_images = None
        with torch.no_grad():
            for i, (_, locations, forces, _, model_images, _, exp_name, traj_props, padding_mask) in enumerate(test_loader):
                traj_lengths = traj_props[:, 5, :].long() if traj_props.shape[1] > 5 else None
                locations, forces, model_images = data_preprocess(
                    locations, forces, model_images, train_mean_locations, train_std_locations,
                    train_mean_forces, train_std_forces, args.dont_norm_locations, args.relative_locations,
                    args.zero_location, args.zero_forces, args.shuffle_order, args.dataset.num_training_trajs,
                    rep_model.device, traj_lengths=traj_lengths)
                padding_mask = padding_mask.to(rep_model.device)

                representation, rep_loss, rep_loss_info, rep_inference_results = rep_model(forces, locations,
                                                                                            padding_mask=padding_mask)
                if args.models.downstream_model.name == "PhantomIndexClassifier":
                    downstream_kwargs = {"exp_name": exp_name}
                elif args.models.downstream_model.name in ["TransposedConvParRepPred", "TransposedConvParRepPred3D"]:
                    downstream_kwargs = {
                        "particles_locations": rep_inference_results["particles_locations"],
                        "active_particles_mask": rep_inference_results["active_particles_mask"],
                        "img_coords": img_coords
                    }
                else:
                    downstream_kwargs = {}
                if args.zero_representation:
                    representation = torch.zeros_like(representation)

                downstream_loss = torch.tensor(0.0, device=rep_model.device)
                if downstream_model is not None:
                    downstream_prediction, downstream_loss, downstream_loss_info, downstream_results = downstream_model(
                        representation, model_images,
                        is_train=False,
                        **downstream_kwargs)

                if i == vis_batch_index:
                    vis_forces = forces.clone()
                    vis_rep_inference_results = rep_inference_results
                    if downstream_model is not None:
                        vis_downstream_results = {k: v.clone() for k, v in downstream_results.items()}
                        vis_model_images = model_images.clone()

                total_loss = args.optimizer.force_reconstruction_weight * rep_loss + args.optimizer.downstream_loss_weight * downstream_loss

                test_loss += total_loss.item()
                for key, value in rep_loss_info.items():
                    if key not in test_rep_losses:
                        test_rep_losses[key] = 0
                    test_rep_losses[key] += value.item()

                test_downstream_loss += downstream_loss.item()
                test_rep_loss += rep_loss.item()

        average_test_loss = test_loss / len(test_loader)
        for key, value in test_rep_losses.items():
            test_rep_losses[key] = value / len(test_loader)

        average_test_downstream_loss = test_downstream_loss / len(test_loader)
        average_test_rep_loss = test_rep_loss / len(test_loader)

        time_loggings["eval total time"] = time.time() - eval_start_time

        # Check if current model is the best based on losses
        current_rep_loss = average_test_rep_loss
        current_downstream_loss = average_test_downstream_loss

        if current_rep_loss < best_rep_loss:
            best_rep_loss = current_rep_loss
            best_rep_model_state = copy.deepcopy(rep_model.state_dict())
            print(f"New best representation loss: {best_rep_loss:.4f} at epoch {epoch + 1}")

        if current_downstream_loss < best_downstream_loss:
            best_downstream_loss = current_downstream_loss
            if downstream_model is not None:
                best_downstream_model_state = copy.deepcopy(downstream_model.state_dict())
                best_rep_model_for_best_downstream_state = copy.deepcopy(rep_model.state_dict())
            print(f"New best downstream loss: {best_downstream_loss:.4f} at epoch {epoch + 1}")

        # Save regular checkpoints
        if args.model_log_interval != -1 and (epoch + 1) % args.model_log_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"rep_model_epoch_{epoch + 1}.pt")
            torch.save(rep_model.state_dict(), checkpoint_path)
            checkpoint_path = os.path.join(checkpoint_dir, f"downstream_model_epoch_{epoch + 1}.pt")
            if downstream_model is not None:
                torch.save(downstream_model.state_dict(), checkpoint_path)
            print(f"Saved model checkpoint at epoch {epoch + 1}")

        wandb.log({"Learning Rate": optimizer.param_groups[0]['lr']}, step=epoch)
        scheduler.step()
        epoch_end_time = time.time()
        time_loggings["epoch total time"] = epoch_end_time - epoch_start_time
        print(
            f'Epoch {epoch + 1}, Train Loss: {average_train_loss:.4f}, Test Loss: {average_test_loss:.4f}')
        wandb.log({"Epoch": epoch, "Train Loss": average_train_loss, "Test Loss": average_test_loss}, step=epoch)
        for key, value in train_rep_losses.items():
            wandb.log({f"Train {key}": value}, step=epoch)
        for key, value in test_rep_losses.items():
            wandb.log({f"Test {key}": value}, step=epoch)

        wandb.log({"Train Downstream Loss": average_train_downstream_loss,
                   "Test Downstream Loss": average_test_downstream_loss}, step=epoch)
        wandb.log({"Train Grads Max": train_grads_max, "Train Grads Average": average_train_grads}, step=epoch)

        # Log best losses
        wandb.log({
            "Best Representation Loss": best_rep_loss,
            "Best Downstream Loss": best_downstream_loss
        }, step=epoch)
        time_loggings_wandb = {f"time/{key}": value for key, value in time_loggings.items()}
        wandb.log(time_loggings_wandb, step=epoch)

        # Log metrics
        if downstream_model is not None:
            log_downstream_models(downstream_model, epoch)

        if args.vis_log_interval != -1 and ((epoch + 1) % args.vis_log_interval == 0 or epoch == 0):
            if downstream_model is not None:
                downstream_average_metrics = downstream_model.get_avg_metrics()
                if "confusion_matrix" in downstream_average_metrics:
                    log_conf_matrix(downstream_average_metrics["confusion_matrix"], epoch)

            if "predicted_forces" in vis_rep_inference_results and "reconstruction_steps" in vis_rep_inference_results and \
                    "input_steps" in vis_rep_inference_results and rep_model.name not in ["LocalReconstructionModel",
                                                                                           "ParticlesReconstructionModel"]:
                log_force_prediction(args.dataset.real_data, vis_rep_inference_results["predicted_forces"],
                                     vis_forces, vis_rep_inference_results["reconstruction_steps"], epoch)
                log_forces_error(vis_rep_inference_results["predicted_forces"], vis_forces,
                                 vis_rep_inference_results["reconstruction_steps"], vis_rep_inference_results["input_steps"],
                                 epoch)

            if downstream_model is not None:
                downstream_vis = downstream_model.visualize_predictions(vis_downstream_results, vis_model_images)
                if downstream_vis is not None:
                    wandb.log({"Predictions": wandb.Image(downstream_vis)}, step=epoch)
                    plt.close(downstream_vis)

    # Save best models at the end of training
    if best_rep_model_state is not None:
        best_rep_checkpoint_path = os.path.join(checkpoint_dir, "best_rep_model.pt")
        torch.save(best_rep_model_state, best_rep_checkpoint_path)

    if best_downstream_model_state is not None:
        best_downstream_checkpoint_path = os.path.join(checkpoint_dir, "best_downstream_model.pt")
        torch.save(best_downstream_model_state, best_downstream_checkpoint_path)
    if best_rep_model_for_best_downstream_state is not None:
        rep_for_best_downstream_path = os.path.join(checkpoint_dir, "rep_model_for_best_downstream.pt")
        torch.save(best_rep_model_for_best_downstream_state, rep_for_best_downstream_path)

    print(f"Best representation loss: {best_rep_loss:.4f}")
    print(f"Best downstream loss: {best_downstream_loss:.4f}")

    wandb.finish()


if __name__ == "__main__":
    train()
