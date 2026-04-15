from src.data_loader import load_data, build_vocab, encode, decode

text = load_data()
print(text[:100])

stoi, itos = build_vocab(text)

encoded = encode(text, stoi)
decoded = decode(encoded, itos)

print(decoded[:200])