# Value target scale intervention

One-factor training intervention: replace terminal targets z with0.1*z, preserve signs and original outcomes. Exact fresh_data4935sampledpositions (3590unique), same indices/shuffle62016, initialstep0,78Adamupdates1e-4 withoutbiascorrection, legalsofttargetCE+valueMSE+L2, independent spatialtrainingBN. EffectiveLR reset after state restore and asserted. No freshgames. Save checkpoints1/8/32/78 and original/scaledtargets.

Search full learned policy and value with no output substitution/rescaling. Exact terminal values remain ±1/0. Both64simulations, independent spatialBN/fixedshape16;32game reused seed162016 bank. Reference fullscale fresh_data2W6D24L15.625%. Scaling changes the value objective/calibration deliberately; no claim that labels were wrong. Track original-outcome and scaled-target validationMSE separately. Oracle masks/states/labels verified before scaling, signs/scaling checked after. Discovery only, no promotion; initial superiority needs independent confirmation and sustainedselfplay.

Motivation: freshbank zeroing only the trained value recovered18.75% to45.3125%, pairedgain26.5625pointsCI[16.015625,37.109375]; no initialsuperiority. Test whether attenuated outcome targets avoid harmful value estimates while retaining outcome learning. Research branch only; PR29 unchanged.
