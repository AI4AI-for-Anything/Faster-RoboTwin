"""Multi-session Pi_05 policy server: ONE server process per GPU serves several eval jobs at once.

Every request frame carries its client's evaluation_id (unique per eval job). Each evaluation_id
gets its own session: a shallow copy of the loaded official Model whose openpi Policy copy has its
own JAX random-number chain starting at key(0) (Policy.__init__ default) and its own observation
buffer. Parameters and the jitted sample function are shared, so a session computes exactly what a
freshly started single-job server would. Model calls stay serialized by the official server's
single model lock. Request ids are uuid4 per request, so the shared response cache cannot mix
sessions. Timing wrappers come from prof_server.py (PROF_SERVER_OUT must be set).

usage: same args as XPolicyLab/setup_policy_server.py
"""
import asyncio
import contextvars
import copy
import json
import time

import prof_server as PS  # noqa: F401  (installs the timers, opens PROF_SERVER_OUT)

from client_server.ws import model_server as WS

_SESSION = contextvars.ContextVar("evalopt_session", default=None)


class SessionRouter:
    """Stands in for server.model; attribute lookups go to the current frame's session."""

    def __init__(self, base):
        self._base = base
        self._sessions = {}

    def session(self, key):
        m = self._sessions.get(key)
        if m is None:
            import jax

            m = copy.copy(self._base)
            m.policy = copy.copy(self._base.policy)
            m.model = m.policy
            m.policy._rng = jax.random.key(0)
            m.observation_window = None
            m._latest_env_idx_list = [0]
            self._sessions[key] = m
            PS.LOG.write(json.dumps({"n": "session_new", "t": round(time.perf_counter() - PS.T0, 4),
                                     "d": 0.0, "sessions": len(self._sessions)}) + "\n")
        return m

    def __getattr__(self, name):
        s = _SESSION.get()
        if s is None:
            raise AttributeError(name)
        return getattr(s, name)


class MultiSessionServer(WS.PolicyServer):
    async def _dispatch_frame(self, frame):
        # runs inside the per-request task, so the context variable is private to this request
        _SESSION.set(self.model.session(frame.evaluation_id))
        return await super()._dispatch_frame(frame)


def main(cfg):
    model = PS.M.Model(cfg)
    server = MultiSessionServer(
        SessionRouter(model),
        WS.PolicyServerConfig(host=cfg.get("host", "0.0.0.0"), port=int(cfg["port"]),
                              ws_ping_interval_s=cfg.get("ws_ping_interval_s", 20.0),
                              ws_ping_timeout_s=cfg.get("ws_ping_timeout_s", 20.0)),
    )
    asyncio.run(server.serve_forever())


if __name__ == "__main__":
    main(PS.S.parse_args_and_config())
