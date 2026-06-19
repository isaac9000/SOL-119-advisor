def get_inputs(axes_and_scalars: dict, device, seed=None):
    import torch
    bs      = axes_and_scalars["batch_size"]
    seq_q   = axes_and_scalars["seq_len_q"]
    seq_kv  = axes_and_scalars["seq_len_kv"]
    dropout = 0.1

    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)

    grad_attn_output     = torch.randn(bs, seq_q, 80, 128, dtype=torch.bfloat16, device=device, generator=g)
    attn_scores_raw      = torch.randn(bs, 80, seq_q, seq_kv, dtype=torch.float32, device=device, generator=g)
    attn_weights         = torch.softmax(attn_scores_raw, dim=-1).to(torch.bfloat16)
    dropout_mask         = torch.rand(bs, 80, seq_q, seq_kv, device=device, generator=g) > dropout
    attn_weights_dropped = (attn_weights.float() * dropout_mask / (1.0 - dropout)).to(torch.bfloat16)
    value_states         = torch.randn(bs, 8, seq_kv, 128, dtype=torch.bfloat16, device=device, generator=g)

    return {
        "grad_attn_output":     grad_attn_output,
        "attn_weights":         attn_weights,
        "attn_weights_dropped": attn_weights_dropped,
        "value_states":         value_states,
        "dropout_mask":         dropout_mask,
        "attention_dropout":    dropout,
    }


@torch.no_grad()
def run(grad_attn_output, attn_weights, attn_weights_dropped,
        value_states, dropout_mask, attention_dropout):
    return custom_kernel((grad_attn_output, attn_weights, attn_weights_dropped,
                          value_states, dropout_mask, attention_dropout))
