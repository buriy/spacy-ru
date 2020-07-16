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


class TqdmUpTo(tqdm):
    """Alternative Class-based version of the above.
    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    Inspired by [twine#242](https://github.com/pypa/twine/pull/242),
    [here](https://github.com/pypa/twine/commit/42e55e06).
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize
