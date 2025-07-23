def forward(
    self,
    hidden_states,           # Tensor passed from HF model
    attention_mask=None,     # Safe default
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=True,
    **kwargs                 # Future-proof: avoids crashing on extra HF args
):
    b, s, _ = hidden_states.shape
    q = self.q(hidden_states).view(b, s, self.h, self.d).transpose(1, 2)
    k_new = self.k(hidden_states).view(b, s, self.h, self.d).transpose(1, 2)
    v_new = self.v(hidden_states).view(b, s, self.h, self.d).transpose(1, 2)

    if use_cache:
        self.cache.prefetch(self.layer, hidden_states.device)

        # Only fetch if exists
        if self.layer in self.cache.store:
            k_hist, v_hist = self.cache.fetch(self.layer, hidden_states.device)
            k_cat = torch.cat([k_hist, k_new], dim=2)
            v_cat = torch.cat([v_hist, v_new], dim=2)
        else:
            k_cat, v_cat = k_new, v_new

        self.cache.save(self.layer, k_cat, v_cat)
    else:
        k_cat, v_cat = k_new, v_new

    scores = torch.matmul(q, k_cat.transpose(-1, -2)) / (self.d ** 0.5)
    probs = torch.softmax(scores, dim=-1)
    y = torch.matmul(probs, v_cat)

    y = y.transpose(1, 2).contiguous().view(b, s, self.h * self.d)
    output = self.o(y)

    # Standards-compliant output
    outputs = (output,)
    if use_cache:
        # HF expects `present_key_value` for future use even if unused
        outputs += ((k_cat, v_cat),)
    if output_attentions:
        outputs += (probs,)
    return outputs
