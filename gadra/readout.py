"""Exact affine diagnostic recipes, recorded alongside captured checkpoints."""


def affine_recipe(model, count, centered=False):
    if model not in ('SceneSplat', 'Chorus') or count < 1:
        raise ValueError('Unknown affine readout')
    if model == 'SceneSplat' and (count != 1 or centered):
        raise ValueError('SceneSplat normalizes one checkpoint feature')
    # Capture stores the current feature first. Chorus concatenates earlier
    # stage endpoints before it; concatenation order changes the affine map.
    order = list(range(1, count)) + [0] if model == 'Chorus' else [0]
    return dict(version=1, model=model, input_order=order, centered=centered)


def validate_recipe(cp, model):
    recipe = cp.get('readout_recipe', {})
    expected = affine_recipe(model, len(cp['readout_inputs']), recipe.get('centered', False))
    if recipe != expected:
        raise ValueError('Readout recipe does not match its captured inputs')
    if sum(s['shape'][-1] for s in cp['readout_inputs']) + 1 != cp['readout_shape'][0]:
        raise ValueError('Affine readout width does not match its real inputs')
    verified = cp.get('readout_recipe_verification', {})
    if not (verified.get('max_abs_error', float('inf')) <= 2e-6
            and verified.get('class_agreement', 0) == 1):
        raise ValueError(f'Readout recipe requires a successful numerical capture: {verified}')
    return recipe


def affine_source(parts, recipe):
    """Real normalized, bias-augmented input to the frozen affine readout."""
    import torch
    from torch.nn import functional as F
    value = torch.cat([parts[i].float() for i in recipe['input_order']], dim=1)
    if recipe['centered']:
        value = value - value.mean(1, keepdim=True)
    value = F.normalize(value, dim=1)
    return F.pad(value, (0, 1), value=1)


def affine_language(parts, weight, recipe, output_fp16=False):
    """Independent recipe interpreter used only to verify capture semantics."""
    from torch.nn import functional as F
    source=affine_source(parts, recipe)
    projected=(source.half()@weight.half()).float() if output_fp16 else source@weight
    return F.normalize(projected,dim=1)


def recipe_check(reference, checked, text):
    """Batchable exact-class verification, including non-power-of-two sizes."""
    import torch
    agrees = (checked @ text.T).argmax(1) == (reference @ text.T).argmax(1)
    # A float32 mean of all ones can round below 1 for real fragment sizes.
    # Integer counting then float64 division also preserves a single mismatch.
    agreement = agrees.sum().double() / len(agrees)
    return torch.stack([(checked - reference).abs().max(), agreement])
