# Documentation Index

This documentation is tied to the current implementation in `src/` and `configs/`.

Read in this order:
1. `docs/pipeline.md` for lifecycle and data/control flow.
2. `docs/training_workflow.md` for practical train/eval iteration commands.
3. `docs/math.md` for MDP and equation-level formulation.
4. `docs/design_choices.md` for why each major design decision was made.
5. `docs/file_reference.md` for file-by-file implementation mapping.

If you are updating the model or environment, update both:
- equations/assumptions in `docs/math.md`,
- parameter/architecture reasoning in `docs/design_choices.md`.
