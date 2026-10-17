# Mirror-paired training

Prior legal-loss matched-normalization checkpoints regress progressively: step8
37.5%, step32 21.875%, step78 12.5% on the discovery bank. Earlier frozen output
substitution implicated learned value. Test color/rank mirror augmentation during
training, distinct from earlier inference-only value averaging which did not fix strength.

Same original4935positions, shuffle62016,78Adamupdates at1e-4, legalCE/valueMSE/L2,
independent spatial training normalization and unchanged independent padded16 search.
Each64-position minibatch gains its exact color/rank mirrors (128 total, mean loss).
Original sample order and update count fixed; doubled positions is the augmentation
intervention. Side-to-move value labels unchanged; policy/masks reflected including
underpromotions. Running-stat updates suppressed. Original and mirrored all19planes,
legal masks, target support and labels independently replay-verified before training.
No fresh games; save all checkpoints1/8/32/78. Same32game seed162016 discovery bank.
Unaugmented reference0W8D24L12.5%. Gains need fresh confirmation and sustained learning.
No production defaults changed; PR29 remains untouched.
