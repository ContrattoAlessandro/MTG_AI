"""The policy inference service: a recurrent policy with idempotent player/game sessions.

Endpoints: POST /act (choose a candidate), /record (store XMage's AI decision
for behaviour cloning), /finish (end a session and write its trajectory),
/reload (opt-in checkpoint swap); GET /healthz, /catalog, /stats. See
docs/policy-protocol.md.

Training labels enter only /finish, after action selection has completed. They
are written to trajectories, never forwarded to the actor. /act is search-free.
"""

import hashlib
import json
import os
import queue
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import torch
from flask import Flask, jsonify, request

from .checkpoint import architecture_name, load_checkpoint
from .model import PolicyMemory
from .roles import ROLE_NAMES
from .schema import SCHEMA_VERSION, collate_observations, validate_labels, validate_observation


@dataclass
class Session:
    """One player's game: recurrent memory, the last answered step, and its recorded decisions."""

    memory: PolicyMemory | None = None
    step: int = -1
    event_cursor: int = -1
    fingerprint: str = ""
    response: dict | None = None
    records: list[dict] = field(default_factory=list)
    sampled: bool = True
    accessed: float = field(default_factory=time.monotonic)
    inflight: "_Job | None" = None
    # Decisions made by XMage's AI and posted to /record (behaviour-cloning data).
    teacher: bool = False


# Episodes of teacher sessions carry this policy_version instead of a checkpoint's.
TEACHER_VERSION = "xmage-teacher"


@dataclass
class _Job:
    """A decision waiting for the inference worker."""

    key: tuple[str, str]
    session: Session
    observation: dict
    digest: str
    new_session: bool
    payload_json: str = ""
    done: threading.Event = field(default_factory=threading.Event)
    response: dict | None = None
    error: BaseException | None = None

    def wait(self) -> dict:
        self.done.wait()
        if self.error is not None:
            raise self.error
        return self.response


def canonical_json(value: dict) -> str:
    """Key-sorted, compact JSON: the basis of retry fingerprints."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: dict) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


class PolicyService:
    """Session bookkeeping in request threads; inference in one batching worker.

    The worker takes every request queued while it was busy (up to max_batch) and
    runs them as one forward pass. There is no wait timer: an idle service answers
    a lone request immediately, and batches grow only under concurrent load.
    """

    def __init__(
        self,
        model,
        catalog,
        policy_version: str,
        output_dir=None,
        max_sessions: int = 64,
        session_ttl: float = 3600,
        max_steps: int = 10000,
        max_batch: int = 32,
        include_belief: bool = True,
    ):
        if isinstance(max_batch, bool) or not isinstance(max_batch, int) or max_batch < 1:
            raise ValueError("max_batch must be a positive integer")
        self.model = model.eval()
        self.catalog = catalog
        self.policy_version = policy_version
        self.output_dir = Path(output_dir) if output_dir else None
        self.max_sessions, self.session_ttl, self.max_steps = max_sessions, session_ttl, max_steps
        self.max_batch = max_batch
        # Belief marginals are diagnostics (~97% of the response bytes); engines
        # that only read action_id can disable them to save GIL-bound JSON work.
        self.include_belief = include_belief
        self.sessions: dict[tuple[str, str], Session] = {}
        self.completed: OrderedDict[tuple[str, str], tuple[str, dict]] = OrderedDict()
        self.lock = threading.RLock()
        self.queue: queue.SimpleQueue[_Job] = queue.SimpleQueue()
        self.started = time.monotonic()
        self.counters = {
            "act_requests": 0,
            "inferences": 0,
            "batches": 0,
            "max_batch": 0,
            "act_seconds": 0.0,
            "inference_seconds": 0.0,
            "first_act": None,
            "last_act": None,
        }
        # Per-player game outcomes reported to /finish since start or last reload.
        self.results = dict.fromkeys(("win", "loss", "draw", "truncated", "aborted"), 0)
        threading.Thread(target=self._worker, name="policy-inference", daemon=True).start()

    def stats(self) -> dict:
        """Request and inference counters, and the game outcomes since start or reload."""
        with self.lock:
            stats = dict(self.counters)
            stats["results"] = dict(self.results)
        for key in ("first_act", "last_act"):
            if stats[key] is not None:
                stats[key] -= self.started  # seconds since service start
        return stats

    def act(self, payload: dict) -> dict:
        """Choose a candidate for {observation, sample}; retries of a decision are idempotent."""
        started = time.monotonic()
        try:
            return self._act(payload)
        finally:
            finished = time.monotonic()
            with self.lock:
                c = self.counters
                c["act_requests"] += 1
                c["act_seconds"] += finished - started
                c["first_act"] = started if c["first_act"] is None else c["first_act"]
                c["last_act"] = finished

    def _act(self, payload: dict) -> dict:
        if not isinstance(payload, dict) or set(payload) != {"observation", "sample"}:
            raise ValueError("/act requires observation and a boolean sample flag")
        if not isinstance(payload["sample"], bool):
            raise ValueError("sample must be a boolean")
        # Validation is pure, so it runs before taking the session lock.
        obs = validate_observation(payload["observation"], self.model.config)
        key = (obs["game_id"], obs["player_id"])
        # The canonical text is both the retry fingerprint and an immutable
        # snapshot for recording, so callers mutating their dict later is harmless.
        payload_json = canonical_json(payload)
        digest = hashlib.sha256(payload_json.encode()).hexdigest()
        with self.lock:
            now = time.monotonic()
            for expired in [
                k for k, v in self.sessions.items() if v.inflight is None and now - v.accessed > self.session_ttl
            ]:
                del self.sessions[expired]
            if key in self.completed:
                raise ValueError("this game/player session is already finished; game IDs must be unique")
            new_session = key not in self.sessions
            if new_session:
                if obs["step"] != 0:
                    raise ValueError("new or expired sessions must begin at step zero")
                if len(self.sessions) >= self.max_sessions:
                    raise ValueError("active session limit reached; finish or abort completed games")
                session = Session(sampled=payload["sample"])
            else:
                session = self.sessions[key]
                if session.teacher:
                    raise ValueError("this session records teacher decisions; it cannot request policy actions")
            session.accessed = now
            pending = session.inflight
            if pending is not None:
                # A retry of the decision still being computed shares its result.
                if obs["step"] != pending.observation["step"] or digest != pending.digest:
                    raise ValueError("a different decision is still in flight for this player")
            elif obs["step"] == session.step:
                if digest != session.fingerprint:
                    raise ValueError("retry changed the observation or sampling mode")
                return session.response
            else:
                if obs["step"] != session.step + 1 or obs["step"] >= self.max_steps:
                    raise ValueError("player steps must be contiguous and within the configured limit")
                if payload["sample"] != session.sampled:
                    raise ValueError("sampling mode cannot change during a recurrent game")
                if obs["events"] and obs["events"][0]["seq"] <= session.event_cursor:
                    raise ValueError("history contains an event already processed by the memory")
                pending = session.inflight = _Job(key, session, obs, digest, new_session, payload_json)
                self.sessions[key] = session
                self.queue.put(pending)
        return pending.wait()

    def record(self, payload: dict) -> dict:
        """Store a decision XMage's AI made among the observation's candidates; no inference.

        Same session rules as /act (step zero, contiguous steps, no replayed events,
        idempotent retries); /finish then writes the episode with TEACHER_VERSION.
        """
        if not isinstance(payload, dict) or set(payload) != {"observation", "action_id"}:
            raise ValueError("/record requires observation and action_id")
        if self.output_dir is None:
            raise ValueError("teacher recording needs an episode directory (serve --episodes)")
        obs = validate_observation(payload["observation"], self.model.config)
        if payload["action_id"] not in {action["id"] for action in obs["actions"]}:
            raise ValueError("recorded teacher action is not one of the observation's candidates")
        key = (obs["game_id"], obs["player_id"])
        digest = fingerprint(payload)
        with self.lock:
            if key in self.completed:
                raise ValueError("this game/player session is already finished; game IDs must be unique")
            session = self.sessions.get(key)
            if session is None:
                if obs["step"] != 0:
                    raise ValueError("new or expired sessions must begin at step zero")
                if len(self.sessions) >= self.max_sessions:
                    raise ValueError("active session limit reached; finish or abort completed games")
                session = Session(teacher=True)
            elif not session.teacher:
                raise ValueError("this session plays the policy; it cannot record teacher decisions")
            session.accessed = time.monotonic()
            if obs["step"] == session.step:
                if digest != session.fingerprint:
                    raise ValueError("retry changed the recorded decision")
                return session.response
            if obs["step"] != session.step + 1 or obs["step"] >= self.max_steps:
                raise ValueError("player steps must be contiguous and within the configured limit")
            if obs["events"] and obs["events"][0]["seq"] <= session.event_cursor:
                raise ValueError("history contains an event already recorded")
            session.records.append(
                {
                    "payload_json": canonical_json({"observation": obs}),
                    "action_id": payload["action_id"],
                    "log_probability": 0.0,
                }
            )
            session.step, session.fingerprint = obs["step"], digest
            session.response = {"schema_version": SCHEMA_VERSION, "recorded_step": obs["step"]}
            if obs["events"]:
                session.event_cursor = obs["events"][-1]["seq"]
            self.sessions[key] = session
            return session.response

    def _worker(self):
        while True:
            jobs = [self.queue.get()]
            while len(jobs) < self.max_batch:
                try:
                    jobs.append(self.queue.get_nowait())
                except queue.Empty:
                    break
            try:
                started = time.monotonic()
                responses, memories = self._infer(jobs)
                elapsed = time.monotonic() - started
            except Exception as error:  # every waiting request must be released
                with self.lock:
                    for job in jobs:
                        job.session.inflight = None
                        if job.new_session and self.sessions.get(job.key) is job.session:
                            del self.sessions[job.key]
                        job.error = error
                        job.done.set()
                continue
            with self.lock:
                c = self.counters
                c["inferences"] += len(jobs)
                c["batches"] += 1
                c["max_batch"] = max(c["max_batch"], len(jobs))
                c["inference_seconds"] += elapsed
                for job, response, memory in zip(jobs, responses, memories):
                    self._commit(job, response, memory)
                    job.response = response
                    job.done.set()

    def _infer(self, jobs: list[_Job]) -> tuple[list[dict], list[PolicyMemory]]:
        device = next(self.model.parameters()).device
        observations = [job.observation for job in jobs]
        batch = collate_observations(observations, self.model.config, validate=False).to(device)
        initial = self.model.initial_memory(1, device)
        previous = [job.session.memory or initial for job in jobs]
        memory = PolicyMemory(torch.cat([m.events for m in previous]), torch.cat([m.decisions for m in previous]))
        with torch.inference_mode():
            out = self.model(batch, memory)
            distribution = out.distribution()
            sampled = torch.tensor([job.session.sampled for job in jobs], device=device)
            actions = torch.where(sampled, distribution.sample(), out.logits.argmax(-1))
            log_probability = distribution.log_prob(actions).cpu().tolist()
            probabilities = distribution.probs.cpu()
            values = out.value.cpu().tolist()
            if self.include_belief:
                count_probabilities = out.belief.probabilities.cpu()
                expected_counts = out.belief.expected_counts.cpu()
                roles = (
                    None
                    if out.role_presence is None
                    else (out.role_presence.cpu().tolist(), out.role_expected_counts.cpu().tolist())
                )
            actions = actions.cpu().tolist()
            # Clone rows so each session keeps only its own state, not the batch.
            memories = [
                PolicyMemory(out.memory.events[i : i + 1].clone(), out.memory.decisions[i : i + 1].clone())
                for i in range(len(jobs))
            ]
        responses = []
        for i, obs in enumerate(observations):
            kinds = len(obs["opponent_deck"])
            response = {
                "schema_version": SCHEMA_VERSION,
                "policy_version": self.policy_version,
                "action_id": obs["actions"][actions[i]]["id"],
                "log_probability": float(log_probability[i]),
                "value": float(values[i]),
                "probabilities": probabilities[i, : len(obs["actions"])].tolist(),
            }
            if not self.include_belief:
                responses.append(response)
                continue
            response["belief"] = {
                "card_ids": [c["card_id"] for c in obs["opponent_deck"]],
                "count_probabilities": count_probabilities[i, :kinds].tolist(),
                "expected_counts": expected_counts[i, :kinds].tolist(),
                "mode": self.model.config.belief_mode,
            }
            if roles is not None:
                response["belief"]["roles"] = {
                    name: {"probability_at_least_one": float(roles[0][i][r]), "expected_count": float(roles[1][i][r])}
                    for r, name in enumerate(ROLE_NAMES)
                }
            responses.append(response)
        return responses, memories

    def _commit(self, job: _Job, response: dict, memory: PolicyMemory):
        session, obs = job.session, job.observation
        if self.output_dir and session.sampled:
            # Parsed back only at /finish, off the per-decision path.
            session.records.append(
                {
                    "payload_json": job.payload_json,
                    "action_id": response["action_id"],
                    "log_probability": response["log_probability"],
                }
            )
        session.memory = memory
        session.step, session.fingerprint, session.response = obs["step"], job.digest, response
        session.accessed = time.monotonic()
        if obs["events"]:
            session.event_cursor = obs["events"][-1]["seq"]
        session.inflight = None

    def reload(self, model, catalog, policy_version, output_dir=None):
        """Swap checkpoints between collection batches; per-batch counters restart.

        Only allowed with no live sessions, so no queued or in-flight decision can
        see the new weights, and a recurrent game never spans two policies.
        """
        if catalog.digest != self.catalog.digest:
            raise ValueError("reloaded checkpoints must keep the served card catalog")
        with self.lock:
            if self.sessions:
                raise ValueError("cannot reload while games are active; finish or abort them first")
            self.model, self.catalog, self.policy_version = model.eval(), catalog, policy_version
            self.output_dir = Path(output_dir) if output_dir else None
            self.completed.clear()
            self.started = time.monotonic()
            self.counters = {
                key: (None if key in ("first_act", "last_act") else 0 * value) for key, value in self.counters.items()
            }
            self.results = dict.fromkeys(self.results, 0)

    def finish(self, payload: dict) -> dict:
        """End a session; write its trajectory with labels when recording a finished game."""
        allowed = {"game_id", "player_id", "result", "terminal", "labels"}
        # "truncated" is optional: an error-free game stopped by the turn limit.
        if not isinstance(payload, dict) or set(payload) - {"truncated"} != allowed:
            raise ValueError("/finish requires game_id, player_id, result, terminal, and labels")
        truncated = payload.get("truncated", False)
        if not isinstance(truncated, bool) or (truncated and (payload["terminal"] is True or payload["result"] != 0)):
            raise ValueError("truncated must be a boolean; a truncated game is not terminal and has result 0")
        if any(not isinstance(payload[field], str) or not payload[field] for field in ("game_id", "player_id")):
            raise ValueError("game_id and player_id must be nonempty strings")
        if not isinstance(payload["labels"], list):
            raise ValueError("labels must be a list")
        key = (payload["game_id"], payload["player_id"])
        digest = fingerprint(payload)
        if (
            isinstance(payload["result"], bool)
            or payload["result"] not in (-1, 0, 1)
            or not isinstance(payload["terminal"], bool)
        ):
            raise ValueError("result must be -1/0/+1 and terminal must be a boolean")
        with self.lock:
            if key in self.completed:
                old_digest, response = self.completed[key]
                if old_digest != digest:
                    raise ValueError("finish retry changed the completed game")
                return response
            if key not in self.sessions:
                raise ValueError("unknown or expired game/player session")
            session = self.sessions[key]
            if session.inflight is not None:
                raise ValueError("cannot finish while a decision is still in flight")
            path = None
            if self.output_dir and session.sampled and (payload["terminal"] or truncated):
                labels = payload["labels"]
                if not isinstance(labels, list) or len(labels) != len(session.records):
                    raise ValueError("one label record is required for every player decision")
                steps = []
                for i, (record, label) in enumerate(zip(session.records, labels)):
                    observation = json.loads(record["payload_json"])["observation"]
                    validate_labels(label, observation)
                    steps.append(
                        {
                            "observation": observation,
                            "action_id": record["action_id"],
                            "log_probability": record["log_probability"],
                            "labels": label,
                            "reward": payload["result"] if i == len(session.records) - 1 else 0,
                        }
                    )
                version = TEACHER_VERSION if session.teacher else self.policy_version
                episode = {
                    "schema_version": SCHEMA_VERSION,
                    "policy_version": version,
                    "steps": steps,
                    "terminal": payload["terminal"],
                    "truncated": truncated,
                }
                self.output_dir.mkdir(parents=True, exist_ok=True)
                path = self.output_dir / f"episode_{uuid.uuid4().hex}.jsonl"
                temporary = path.with_suffix(".tmp")
                try:
                    temporary.write_text(json.dumps(episode, allow_nan=False) + "\n", encoding="utf-8")
                    os.replace(temporary, path)
                finally:
                    if temporary.exists():
                        temporary.unlink()
            # Aborted games release memory and are excluded from PPO data. Truncated
            # games are recorded for bootstrapping; neither is ever called a draw.
            response = {
                "recorded": path is not None,
                "episode_file": str(path) if path else None,
                "terminal": payload["terminal"],
                "truncated": truncated,
            }
            del self.sessions[key]
            outcome = (
                "truncated"
                if truncated
                else "aborted"
                if not payload["terminal"]
                else {1: "win", -1: "loss", 0: "draw"}[payload["result"]]
            )
            self.results[outcome] += 1
            self.completed[key] = (digest, response)
            while len(self.completed) > 1024:
                self.completed.popitem(last=False)
            return response


def create_app(service: PolicyService, allow_reload: bool = False) -> Flask:
    """The Flask application exposing `service` over HTTP."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

    @app.errorhandler(ValueError)
    def invalid(error):
        return jsonify(error=str(error)), 400

    @app.get("/healthz")
    def health():
        return jsonify(
            schema_version=SCHEMA_VERSION,
            policy_version=service.policy_version,
            architecture=architecture_name(service.model.config),
        )

    @app.get("/catalog")
    def catalog():
        return jsonify(catalog=service.catalog.to_dict(), catalog_digest=service.catalog.digest)

    @app.get("/stats")
    def stats():
        return jsonify(service.stats())

    @app.post("/act")
    def act():
        return jsonify(service.act(request.get_json()))

    @app.post("/record")
    def record():
        return jsonify(service.record(request.get_json()))

    @app.post("/finish")
    def finish():
        return jsonify(service.finish(request.get_json()))

    if allow_reload:

        @app.post("/reload")
        def reload():
            # Collection runners keep this process (and its imports) between
            # generations. Opt-in: it loads a checkpoint path named by the caller.
            payload = request.get_json()
            if not isinstance(payload, dict) or set(payload) != {"checkpoint", "episodes"}:
                raise ValueError("/reload requires checkpoint and episodes (null to disable recording)")
            if not isinstance(payload["checkpoint"], str) or not (
                payload["episodes"] is None or isinstance(payload["episodes"], str)
            ):
                raise ValueError("checkpoint must be a path and episodes a path or null")
            device = next(service.model.parameters()).device
            model, _, catalog, data = load_checkpoint(payload["checkpoint"], device)
            service.reload(model, catalog, data["policy_version"], payload["episodes"])
            return jsonify(policy_version=service.policy_version)

    return app


def serve(
    checkpoint,
    host: str = "127.0.0.1",
    port: int = 50052,
    output_dir=None,
    device: str = "auto",
    max_batch: int = 32,
    include_belief: bool = True,
    torch_threads: int = 1,
    allow_reload: bool = False,
) -> None:
    """Serve a checkpoint with waitress until interrupted."""
    import waitress

    torch.set_num_threads(torch_threads)
    model, _, catalog, data = load_checkpoint(checkpoint, device)
    service = PolicyService(
        model, catalog, data["policy_version"], output_dir, max_batch=max_batch, include_belief=include_belief
    )
    print(f"Serving policy {data['policy_version']} on {host}:{port}; device={next(model.parameters()).device}")
    waitress.serve(create_app(service, allow_reload), host=host, port=port, threads=16)
