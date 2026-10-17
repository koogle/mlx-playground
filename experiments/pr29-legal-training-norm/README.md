# Match training/search normalization after legal CE

Frozen shuffled-checkpoint interventions found learned value harmful on the discovery
bank: initialpolicy+trainedvalue1W4D27L9.375%, trainedpolicy+initialvalue14W3D15L48.4375%.
Original batch-dependent versus per-position value outputs differed by up to1.55;
fixed-shape independent inference error was zero. This motivates testing whether
normalization mismatch damages value learning after the legal-policy intervention.

Single new arm: same initial weights,4935positions, shuffle62016, batch64,78Adam
updates without bias correction at1e-4, exact legal-action CE + valueMSE + L2.
Only training normalization changes to per-position spatial statistics with running
updates suppressed. No freezing of trainable affine parameters or gradients.
Search unchanged: independent spatialBN, fixed paddedshape16,64simulations.
Compare32games on same seed162016 bank to shuffled standardtrainingBN9.375%.
Retain checkpoint1/8/32/78 and effectiveLR assertions, exact replay masks and labels.

Earlier matching normalization with global CE did not help. This conditional test
addresses value damage after legal CE, not a claim that normalization is the sole
cause. Discovery only; gains need fresh confirmation and sustained selfplay tests.
