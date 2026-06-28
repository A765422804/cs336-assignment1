from nn import TransformerBlock

tb = TransformerBlock(2, 1, 1, 1, 1)

print(tb.state_dict().keys())