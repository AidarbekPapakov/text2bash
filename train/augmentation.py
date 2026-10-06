"""Training-data augmentation for text2bash. See NOTES.md's error analysis."""
import collections


def toolname(command):
    command = command.strip()
    if command.startswith("sudo "):
        command = command[5:]
    parts = command.split()
    return parts[0] if parts else ""


def oversample_rare_tldr(records, factor=3, threshold=2):
    """Duplicate tldr examples whose tool appears <= threshold times in training.

    tldr covers thousands of distinct CLI tools with only a handful of examples
    each; the model needs more repetition of these to learn them instead of
    hallucinating a plausible-but-wrong tool (see NOTES.md error analysis).
    """
    tldr = [r for r in records if r.get("source") == "tldr"]
    freq = collections.Counter(toolname(r["command"]) for r in tldr)
    augmented = list(records)
    for r in tldr:
        if freq[toolname(r["command"])] <= threshold:
            augmented.extend([r] * (factor - 1))
    print(f"Oversampled rare tldr examples: {len(augmented) - len(records)} duplicates added "
          f"({len(records)} -> {len(augmented)} total training examples)")
    return augmented
