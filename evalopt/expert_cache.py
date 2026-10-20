"""Record / replay the expert-check outcome of every seed, leaving the upstream eval loop untouched.

The official loop runs, per raw seed: setup_demo -> play_once -> close_env -> check_success (the
"expert phase"), and only for accepted seeds a second setup_demo + policy rollout (the "eval phase").
The expert phase does not involve the policy. This module wraps the task-env class methods:

  record (EVALOPT_EXPERT_RECORD=<file.pkl>): run everything for real and store, per seed, how the
      expert phase ended (setup unstable / setup error / play error / plan_success, check_success)
      and the dict play_once returned (it feeds build_instruction).
  replay (EVALOPT_EXPERT_REPLAY=<file.pkl>): in the expert phase do no simulation at all; re-raise
      the recorded exception class, or return the recorded play_once dict and check_success value.
      The eval phase (second setup_demo and everything after) runs for real.

Phase per seed = order of setup_demo calls for that seed (1st = expert, 2nd = eval), exactly as the
official loop issues them with expert_check=true.
"""
import copy
import os
import pickle

from envs.utils.create_actor import UnStableError

RECORD = os.environ.get("EVALOPT_EXPERT_RECORD")
REPLAY = os.environ.get("EVALOPT_EXPERT_REPLAY")


_CURRENT = [None]  # the ExpertCache the installed wrappers use; swapped per job by a resident worker


class ExpertCache:
    def __init__(self, record=None, replay=None):
        self.record_path = record if record is not None else RECORD
        self.replay_path = replay if replay is not None else REPLAY
        _CURRENT[0] = self
        self.calls = {}          # seed -> number of setup_demo calls so far
        self.cur = None          # seed of the current phase
        self.phase = None        # "expert" | "eval"
        self.pending_check = False
        self.records = {}        # seed -> dict
        self.replay = None
        self.replayed = 0
        if self.replay_path:
            with open(self.replay_path, "rb") as f:
                self.replay = pickle.load(f)["seeds"]

    def active(self):
        return bool(self.record_path or self.replay_path)

    def save(self, meta):
        if self.record_path:
            with open(self.record_path, "wb") as f:
                pickle.dump({"meta": meta, "seeds": self.records}, f)

    @staticmethod
    def install(cls):
        if getattr(cls, "_expert_cache_installed", False):
            return
        cls._expert_cache_installed = True
        orig_setup, orig_play = cls.setup_demo, cls.play_once
        orig_check, orig_close = cls.check_success, cls.close_env

        def setup_demo(env, **kw):
            ec = _CURRENT[0]
            if ec is None or not ec.active():
                return orig_setup(env, **kw)
            seed = kw.get("seed")
            ec.calls[seed] = ec.calls.get(seed, 0) + 1
            ec.cur, ec.phase = seed, ("expert" if ec.calls[seed] == 1 else "eval")
            ec.pending_check = False
            if ec.phase == "expert" and ec.replay is not None:
                rec = ec.replay[seed]
                ec.replayed += 1
                if rec["setup"] == "unstable":
                    raise UnStableError(f"cached: unstable seed {seed}")
                if rec["setup"] != "ok":
                    raise RuntimeError(f"cached expert setup error: {rec['setup']}")
                return None
            rec = ec.records.setdefault(seed, {"setup": None}) if ec.phase == "expert" else None
            try:
                out = orig_setup(env, **kw)
            except UnStableError:
                if rec is not None:
                    rec["setup"] = "unstable"
                raise
            except Exception as e:
                if rec is not None:
                    rec["setup"] = f"error:{type(e).__name__}"
                raise
            if rec is not None:
                rec["setup"] = "ok"
            return out

        def play_once(env):
            ec = _CURRENT[0]
            if ec is None or not ec.active():
                return orig_play(env)
            seed = ec.cur
            if ec.phase == "expert" and ec.replay is not None:
                rec = ec.replay[seed]
                if rec.get("play") != "ok":
                    raise RuntimeError(f"cached expert play error: {rec.get('play')}")
                env.plan_success = rec["plan_success"]
                ec.pending_check = True
                return copy.deepcopy(rec["info"])
            rec = ec.records.get(seed) if ec.phase == "expert" else None
            try:
                info = orig_play(env)
            except Exception as e:
                if rec is not None:
                    rec["play"] = f"error:{type(e).__name__}"
                raise
            if rec is not None:
                rec["play"] = "ok"
                rec["info"] = copy.deepcopy(info)
                ec.pending_check = True
            return info

        def check_success(env, *a, **k):
            ec = _CURRENT[0]
            if ec is None or not ec.active():
                return orig_check(env, *a, **k)
            # Only the loop's own call right after the expert phase (no arguments) is recorded /
            # replayed; tasks also call check_success(target=...) inside play_once (open_laptop,
            # open_microwave) and every physics substep during the rollout.
            if ec.pending_check and not a and not k:
                ec.pending_check = False
                rec = (ec.replay if ec.replay is not None else ec.records)[ec.cur]
                if ec.replay is not None:
                    return rec["check_success"]
                out = orig_check(env)
                rec["plan_success"] = bool(env.plan_success)
                rec["check_success"] = bool(out)
                return out
            return orig_check(env, *a, **k)

        def close_env(env, clear_cache=False):
            ec = _CURRENT[0]
            if ec is None or not ec.active():
                return orig_close(env, clear_cache=clear_cache)
            if ec.phase == "expert" and ec.replay is not None:
                return None
            return orig_close(env, clear_cache=clear_cache)

        cls.setup_demo, cls.play_once = setup_demo, play_once
        cls.check_success, cls.close_env = check_success, close_env
