"""MageZero: a search-free recurrent entity policy for two-player Magic: The Gathering.

Modules, from observation to training run:

    schema          observation schema, validation and tensor collation
    catalog, roles  the card catalog and its reviewed card roles
    belief          learned constrained hand-count belief (optional mode)
    statistics      the without-replacement hand prior and role statistics
    model           the actor (EntityBeliefPolicy) and the privileged critic
    checkpoint      checkpoint format, loading and saving
    service         the HTTP inference/recording service used by XMage
    service_client  launching and talking to service subprocesses
    engine          the persistent XMage engine and its game-batch configs
    shaping         potential-based reward shaping
    learner         recurrent PPO
    teacher         behaviour cloning from XMage's AI (warm start)
    league          historical self-play opponents
    runner          resumable self-play training runs
    diagnostics     hand-prediction evaluation and decision traces
    cli             the `mz` command
"""

__version__ = "0.2.0"
