from scenarios.s1_synthetic.make_clean import make_clean
from estimators.basic.ipdft import IpDFT

def test_ipdft_smoke():
    fs = 5000
    sig, truth = make_clean(f0=60.0, df=0.1, duration=0.2, fs=fs)
    est = IpDFT(fs=fs, frame_len=256)
    out = [est.update(x) for x in sig]
    assert len(out) == len(sig)
    # rough sanity: mean within a couple of Hz of target (loose bound for now)
    assert abs(sum(out)/len(out) - 60.0) < 2.0