"""Copy of upstream eval_remote_policy (RoboTwin ea8b211, scripts/eval_policy_xpolicylab.py
lines 772-975) with one marked change (EVALOPT lazy obs). Everything else is byte-identical; check
with: diff <(sed -n 772,975p eval_policy_xpolicylab.py) <(sed -n '/^def eval_remote_policy_opt/,$p' opt_loop.py)
"""
import subprocess
import traceback
from typing import Any

from eval_policy_xpolicylab import (build_instruction, is_episode_end, normalize_action_chunk,
                                    notify_trial_end, parse_bool, prepare_policy_case, reset_policy,
                                    robotwin_obs_to_xpolicylab, safe_close_env,
                                    xpolicylab_action_to_robotwin)
from envs.utils.create_actor import UnStableError

POLICY_CAMS = frozenset({"head_camera", "left_camera", "right_camera"})
VIDEO_CAMS = frozenset({"head_camera"})


def get_obs_cams(task_env, cams):
    """task_env.get_obs() with only `cams` rendered (render_only is read by the Camera patch)."""
    task_env.cameras._evalopt_render_only = cams
    try:
        return task_env.get_obs()
    finally:
        task_env.cameras._evalopt_render_only = None


def eval_remote_policy_opt(
    task_name: str,
    task_env,
    args: dict[str, Any],
    model_client,
    st_seed: int,
    *,
    usr_args: dict[str, Any],
    test_num: int = 100,
    video_size: str | None = None,
    instruction_type: str | None = None,
) -> tuple[int, int]:
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = parse_bool(usr_args.get("expert_check", True))
    frequency = int(usr_args.get("frequency", usr_args.get("ctrl_freq", 30)))
    action_type = str(usr_args.get("action_type", "joint"))

    task_env.suc = 0
    task_env.test_num = 0

    now_id = 0
    succ_seed = 0
    now_seed = st_seed
    clear_cache_freq = args["clear_cache_freq"]
    args["eval_mode"] = True
    consecutive_env_errors = 0

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0
        episode_info = {"info": {}}

        if expert_check:
            try:
                task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = task_env.play_once()
                task_env.close_env()
            except UnStableError:
                safe_close_env(task_env)
                print(f"skip unstable seed={now_seed} (expert check)")
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                safe_close_env(task_env)
                now_seed += 1
                args["render_freq"] = render_freq
                print(f"error occurs during expert check! seed={now_seed - 1} err={type(e).__name__}: {e}")
                continue

        if expert_check and not (task_env.plan_success and task_env.check_success()):
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq
        try:
            task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        except UnStableError:
            safe_close_env(task_env)
            print(f"skip unstable seed={now_seed} (eval setup)")
            now_seed += 1
            continue
        except Exception as e:
            safe_close_env(task_env)
            print(f"skip seed={now_seed} setup error: {type(e).__name__}: {e}")
            now_seed += 1
            continue

        succ_seed += 1

        instruction = build_instruction(args, episode_info, instruction_type, test_num, now_seed)
        task_env.set_instruction(instruction=instruction)

        if task_env.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{task_env.eval_video_path}/episode{task_env.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            task_env._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        rollout_steps = 0
        rollout_failed = False
        prepare_policy_case(model_client, task_name, now_seed, instruction, action_type)
        reset_policy(model_client)
        try:
            while not is_episode_end(task_env):
                observation = task_env.get_obs()
                xpl_obs = robotwin_obs_to_xpolicylab(
                    observation,
                    instruction=task_env.get_instruction(),
                    env_idx=0,
                    frequency=frequency,
                    task_env=task_env,
                )
                model_client.call(func_name="update_obs", obs=xpl_obs)
                action_chunk = normalize_action_chunk(model_client.call(func_name="get_action"))
                if len(action_chunk) == 0:
                    raise RuntimeError("Policy returned an empty action chunk.")

                for action_idx, action in enumerate(action_chunk):
                    flat_action, robotwin_action_type = xpolicylab_action_to_robotwin(
                        action,
                        action_type=action_type,
                        current_observation=observation,
                    )
                    task_env.take_action(flat_action, action_type=robotwin_action_type)
                    rollout_steps += 1

                    if task_env.eval_success:
                        succ = True
                        break
                    if is_episode_end(task_env) or action_idx + 1 == len(action_chunk):
                        break

                    # EVALOPT lazy obs: the policy reads only the observation sent right before
                    # get_action, which is always the full get_obs() at the top of the while loop.
                    # Mid-chunk, render only what is consumed (head camera -> now_obs -> eval video)
                    # and keep joint/endpose state for the action fallback; skip update_obs.
                    # With the eval video turned off (EVALOPT_NO_VIDEO) no camera is rendered here.
                    observation = get_obs_cams(
                        task_env, VIDEO_CAMS if task_env.eval_video_path is not None else frozenset()
                    )

                if succ:
                    break
        except Exception:
            rollout_failed = True
            print("\n\033[91mPolicy rollout error:\033[0m")
            print(traceback.format_exc())

        if task_env.eval_video_path is not None:
            task_env._del_eval_video_ffmpeg()

        notify_trial_end(model_client, task_name, now_seed, succ)

        if rollout_failed and rollout_steps == 0 and not succ:
            # Env broke before the policy could act (e.g. renderer buffer errors).
            # Don't burn an episode on it; clear caches and retry with the next seed.
            consecutive_env_errors += 1
            succ_seed -= 1
            safe_close_env(task_env, clear_cache=True)
            print(
                f"env error before first action at seed={now_seed}; episode not counted "
                f"(consecutive={consecutive_env_errors})"
            )
            if consecutive_env_errors >= 5:
                raise RuntimeError(
                    "Aborting task: 5 consecutive env errors before first action "
                    "(simulator/renderer is likely wedged)."
                )
            now_seed += 1
            continue
        consecutive_env_errors = 0

        if succ:
            task_env.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        task_env.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if task_env.render_freq:
            task_env.viewer.close()

        task_env.test_num += 1
        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | "
            f"\033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{task_env.suc}/{task_env.test_num}\033[0m => "
            f"\033[95m{round(task_env.suc / task_env.test_num * 100, 1)}%\033[0m, "
            f"current seed: \033[90m{now_seed}\033[0m\n"
        )
        now_seed += 1

    return now_seed, task_env.suc
