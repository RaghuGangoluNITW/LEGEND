import time, torch, sys
sys.path.insert(0,'.')
from scripts.train_steele_baselines import BrainTopoGCN, EEG_GLT_Net, SAMGCN

device = torch.device('cuda')
C, T, nc, N = 51, 1000, 4, 16441

for model_name, cls, bs in [('BrainTopoGCN', BrainTopoGCN, 32),
                              ('EEG_GLT-Net',  EEG_GLT_Net,  32),
                              ('SAMGCN',        SAMGCN,       32)]:
    model = cls(C, nc, T=T).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit  = torch.nn.CrossEntropyLoss()
    x = torch.randn(bs, C, T, device=device)
    y = torch.randint(0, nc, (bs,), device=device)
    # warmup
    for _ in range(3):
        opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    # timed
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    torch.cuda.synchronize()
    t1 = time.time()
    spb = (t1-t0)/5
    n_b = N // bs
    spe = spb * n_b
    full = spe * 150 * 10 / 3600
    print(f"{model_name:15s} bs={bs} {spb:.3f}s/batch ~{n_b}b/ep => "
          f"{spe:.0f}s/ep  fold@150ep={spe*150/60:.0f}min  "
          f"10-fold={full:.1f}hr  early~{full*0.5:.1f}hr")
    del model; torch.cuda.empty_cache()
