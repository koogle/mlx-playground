# Position-order intervention with legal policy loss

The independent confirmation established less regression with legal CE than global
CE, but legal CE still scored35.9375% against initial (34W24D70L). It is not a fix.
Next test: shuffle the 4935 saved training positions with deterministic seed62016,
keeping initial weights, data, legal-action policy CE, value MSE/L2, batch64,
78Adamupdates without bias correction at1e-4 and normalization settings fixed.
Standard training BN means shuffling also changes batch companions; interpret as
an ordering/batch-composition intervention, not isolated optimizer order.

Search uses independent spatial BN and fixed padding16,64simulations. Evaluate on
the reused32game seed162016 discovery bank against initial, comparing to ordered
legal CE11W6D15L43.75%. Do not interpret a discovery score as confirmed strength.
Retain checkpoints1/8/32/78, effectiveLR assertions, full legal masks and replay
checks, frozen model fingerprints. No fresh selfplay or production defaults changed.

Earlier global-loss shuffle did not help; this experiment tests ordering only after
removing the now-confirmed large regression contribution of global policy loss.
