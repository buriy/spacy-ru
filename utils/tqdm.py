from tqdm.auto import tqdm

def tqdm_batches(batches, total=None, leave=True, **info):
    """
    import time
    dataset = range(50000)
    batches = minibatch(dataset, size=compounding(1., 32., 1.0005))
    for b in tqdm_batches(batches, total=50000, epoch=1):
        time.sleep(0.001)
    """
    infostr = ', '.join([f"{k}={v}" for k,v in info.items()])
    ll = 0
    batch_iter = tqdm(total=total, leave=leave)
    for batch in batches:
        bl = len(batch)
        if bl > ll:
            batch_iter.set_description(f"bsz={bl} "+infostr)
            ll = bl
        yield batch
        batch_iter.update(bl)
    batch_iter.close()
