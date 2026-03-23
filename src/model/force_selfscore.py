# src/model/force_selfscore.py
# -*- coding: utf-8 -*-

import torch
from transformers.generation.logits_process import LogitsProcessor


class ForceSelfScoreGrades(LogitsProcessor):
    """
    Делает self_score строго одним токеном: <grade_1>.. <grade_5>
    и после этого разрешает закрыть кавычку.
    """

    def __init__(self, tokenizer, allowed=None):
        self.tokenizer = tokenizer
        if allowed is None:
            allowed = ["<grade_1>", "<grade_2>", "<grade_3>", "<grade_4>", "<grade_5>"]

        self.allowed_ids = []
        for t in allowed:
            tid = tokenizer.convert_tokens_to_ids(t)
            if tid is None or (hasattr(tokenizer, "unk_token_id") and tid == tokenizer.unk_token_id):
                ids = tokenizer.encode(t, add_special_tokens=False)
                if len(ids) == 1:
                    tid = ids[0]
            if tid is not None and (not hasattr(tokenizer, "unk_token_id") or tid != tokenizer.unk_token_id):
                self.allowed_ids.append(tid)

        if not self.allowed_ids:
            raise RuntimeError("No <grade_*> tokens found in tokenizer vocab.")

        # id для кавычки "
        q = tokenizer.encode('"', add_special_tokens=False)
        self.quote_id = q[0] if len(q) == 1 else None

    def __call__(self, input_ids, scores):
        txt = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)

        marker = '"self_score": "'
        idx = txt.rfind(marker)
        if idx == -1:
            return scores

        after = txt[idx + len(marker):]

        # если кавычка уже встретилась — поле закрыто
        if '"' in after:
            return scores

        # мы в self_score, но проверим: уже сгенерен ли grade?
        # если after уже содержит '<grade_' — значит grade уже есть, теперь разрешаем только закрыть кавычку
        if "<grade_" in after:
            if self.quote_id is None:
                return scores  # fallback
            masked = torch.full_like(scores, float("-inf"))
            masked[:, self.quote_id] = scores[:, self.quote_id]
            return masked

        # иначе grade ещё не было -> разрешаем только grade токены, и запрещаем закрыть кавычку прямо сейчас
        masked = torch.full_like(scores, float("-inf"))
        masked[:, self.allowed_ids] = scores[:, self.allowed_ids]
        if self.quote_id is not None:
            masked[:, self.quote_id] = float("-inf")
        return masked

