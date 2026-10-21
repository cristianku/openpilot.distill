"""Teacher models of the supervised distillation, as paths relative to the repo root.

DEFAULT_TEACHER  "MD v3": commaai/openpilot_driving_models on Hugging Face,
                 1365126a-e71e-4778-be00-8ab92dc82c09/12864/big_driving_supercombo.onnx, the big model that
                 openpilot master runs since commit 1e219d778 "MD v3". Queue-format interface (the history
                 is kept inside the ONNX), split by supervised/onnx_helpers.py split_queue_supercombo.
                 Its hidden_state is 32 spatial tokens x 512 per step.
OLD_TEACHER      the big model of openpilot 084747c75 from GitHub, the teacher of upstream openpilot.distill:
                 kept to resume the runs started with it and for comparisons. Its hidden_state is one
                 pooled 512 vector per step.

helpers/train_supervised.sh --teacher=md-v3|084747c7 downloads them (pinned URL and sha256);
supervised/README.md has the curl commands. The file name says which model it is, so the two never
overwrite each other.

Standard library only: evaluation/ and the web site can import it without torch or onnx.
"""

DEFAULT_TEACHER = "models/big_driving_supercombo_1365126a_12864.onnx"
OLD_TEACHER = "models/big_driving_supercombo.onnx"
