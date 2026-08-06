import sys, time, torch
from sextant.model.transformer import Transformer, ModelArgs
from sextant.seeding import seed_everything, paired_generator

arm, d, mb = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
seq = 1024
seed_everything(1); gen = paired_generator(1)
tw = True if arm == "C" else False if arm == "D" else None
m = Transformer(ModelArgs(arm=arm, d_model=d, vocab_size=50257, depth=8, seq_len=seq, twist=tw), gen).cuda()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
x = torch.randint(0, 50257, (mb, seq)).cuda(); y = torch.randint(0, 50257, (mb, seq)).cuda()
for _ in range(2):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = m(x, y)
    loss.backward(); opt.step()
torch.cuda.synchronize(); t = time.time(); N = 5
for _ in range(N):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = m(x, y)
    loss.backward(); opt.step()
torch.cuda.synchronize(); dt = time.time() - t
tps = N * mb * seq / dt
mem = torch.cuda.max_memory_allocated() / 1e9
print(f"{arm} d={d} mb={mb} tok_s={tps:.0f} min_per_100M={100e6/tps/60:.1f} peakmem_GB={mem:.1f}")
