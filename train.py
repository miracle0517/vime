import logging
import os

import ray

from vime.backends.speculative_training.config import should_run_draft_interval
from vime.ray.placement_group import (
    create_draft_model,
    create_placement_groups,
    create_rollout_manager,
    create_training_models,
)
from vime.utils import logging_utils
from vime.utils.arguments import parse_args
from vime.utils.common import is_npu
from vime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, update_tracking_open_metrics
from vime.utils.metric_utils import compute_rollout_step
from vime.utils.misc import should_run_periodic_action

if is_npu():
    import megatron_adaptor  # noqa: F401

logger = logging.getLogger(__name__)


def _log_draft_result(args, rollout_id, prefix, result):
    if not isinstance(result, dict):
        return
    metrics = {
        f"draft/{prefix}_{key}": value
        for key, value in result.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if not metrics:
        return
    metrics["rollout/step"] = compute_rollout_step(args, rollout_id)
    logging_utils.log(args, metrics, step_key="rollout/step")


def train(args):
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with vLLM engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # Update primary W&B with vLLM metrics endpoint now that servers are up.
    router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
    update_tracking_open_metrics(args, router_addr)

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)
    draft_model = create_draft_model(args, actor_model)
    if args.draft_save_hf:
        logger.info(
            "External DSpark export enabled: template=%s, interval=%s. Artifacts are written on "
            "Actor rank zero; use a shared path when Ray workers run on multiple hosts.",
            args.draft_save_hf,
            args.draft_save_interval,
        )

    if args.offload_rollout:
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

    def save(rollout_id):
        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if actor_trains_this_step:
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))

    # train loop.
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        actor_train_results = None
        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains_this_step:
                actor_train_results = ray.get(
                    actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs)
                )
            else:
                ray.get(value_refs)
        else:
            actor_train_results = ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        draft_snapshot_ref = None
        draft_snapshot_version = None
        if draft_model is not None and actor_trains_this_step and actor_train_results is not None:
            collect_result = draft_model.collect_actor_results(actor_train_results)
            collected_this_rollout = int(collect_result.get("accepted", 0)) > 0
            if collected_this_rollout:
                logger.info("External Draft feature collection: %s", collect_result)
                _log_draft_result(args, rollout_id, "collect", collect_result)
            if collected_this_rollout and should_run_draft_interval(rollout_id, args.draft_train_interval):
                draft_train_result = draft_model.train_draft(rollout_id)
                logger.info("External Draft training: %s", draft_train_result)
                _log_draft_result(args, rollout_id, "train", draft_train_result)
            if should_run_draft_interval(rollout_id, args.draft_publish_interval):
                prepared_snapshot = draft_model.prepare_publish_snapshot()
                if prepared_snapshot is not None:
                    draft_snapshot_ref, draft_snapshot_version = prepared_snapshot

        actor_save_due = should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        )
        if actor_save_due:
            save(rollout_id)

        draft_checkpoint_due = draft_model is not None and (
            (args.draft_save_interval is None and actor_save_due)
            or should_run_draft_interval(rollout_id, args.draft_save_interval)
            or rollout_id == args.num_rollout - 1
        )
        if draft_checkpoint_due:
            draft_save_results = draft_model.save_draft(
                rollout_id,
                export_hf=bool(args.draft_save_hf),
            )
            logger.info("External Draft save completed: %s", draft_save_results)

        offload_train(actor_trains_this_step)
        if args.offload_rollout:
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
