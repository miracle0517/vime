import logging
import os

import ray

from vime.platforms import current_platform

if current_platform().is_npu:
    import vime.backends.megatron_utils  # noqa: F401

from vime.observability.logging_utils import configure_logger, finish_tracking, init_tracking
from vime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from vime.utils.arguments import parse_args
from vime.utils.misc import should_run_periodic_action


def train(args):
    configure_logger()
    release_train = args.release_train

    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with vLLM engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)
    draft_model = create_draft_model(args, actor_model)
    if args.draft_save_hf:
        logger.info(
            "External DSpark export enabled: template=%s, interval=%s. Artifacts are written on "
            "Actor rank zero; use a shared path when Ray workers run on multiple hosts.",
            args.draft_save_hf,
            args.draft_save_interval,
        )

    if args.offload_rollout and not release_train:
        ray.get(rollout_manager.onload_weights.remote())

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(actor_trains_this_step):
        if os.environ.get("VIME_EXTERNAL_DRAFT_SMOKE_SKIP_ACTOR_UPDATE", "0") == "1":
            # The smoke path leaves Actor parameters untouched; avoid exercising
            # unrelated per-rank allocator cleanup before Draft publication.
            return
        # Each model auto-offloads after train() when offload_train is set,
        # so we only need clear_memory for the non-offload case.
        if not args.offload_train:
            if not args.use_critic or actor_trains_this_step:
                actor_model.clear_memory()
            else:
                critic_model.clear_memory()

    # train loop.
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if release_train:
            actor_model.create()

        actor_trains = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            actor_train_results = ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if release_train or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            force_sync = release_train or rollout_id == args.num_rollout - 1
            if actor_trains:
                actor_model.save_model(rollout_id, force_sync=force_sync)
            if args.use_critic:
                critic_model.save_model(rollout_id, force_sync=force_sync)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        offload_train(actor_trains)
        if args.offload_rollout and not release_train:
            ray.get(rollout_manager.onload_weights.remote())
        if draft_snapshot_ref is not None:
            actor_model.set_external_draft_weights(
                draft_snapshot_ref,
                draft_snapshot_version,
            )
        weight_update_results = actor_model.update_weights()
        if draft_snapshot_version is not None:
            if not weight_update_results or not all(value is True for value in weight_update_results):
                raise RuntimeError(
                    f"Draft {draft_snapshot_version} was staged but rollout weight publication did not complete"
                )
            draft_model.mark_published(draft_snapshot_version)
            _log_draft_result(
                args,
                rollout_id,
                "publish",
                {"published": 1, "draft_version": int(draft_snapshot_version)},
            )

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    if draft_model is not None:
        draft_model.release()
    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
