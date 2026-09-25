from transformers import AutoTokenizer

from .intern import InternVLInterface


class MiniInternVL2DADriveLMInterface(InternVLInterface):
    model_label = "Mini-InternVL2-4B-DA-DriveLM"

    def load_tokenizer(self):
        # This checkpoint provides a fast tokenizer path that works with
        # current transformers, while forcing the slow tokenizer hits the
        # missing `vocab_file` path and crashes during service startup.
        return AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True,
        )
