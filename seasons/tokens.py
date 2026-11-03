"""SCS tokens: strings of up to 12 characters packed into a u64 in base 38."""
CHARS = "\0" + "0123456789abcdefghijklmnopqrstuvwxyz_"
_INDEX = {c: i for i, c in enumerate(CHARS)}


def encode(text):
    if len(text) > 12:
        raise ValueError(f"token too long: {text}")
    return sum(_INDEX[c] * 38 ** i for i, c in enumerate(text))


def decode(value):
    out = []
    while value:
        out.append(CHARS[value % 38])
        value //= 38
    return "".join(out)
