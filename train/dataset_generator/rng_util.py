def rng_choice(rng, container):
    i = rng.integers(0, high=len(container))
    return container[i]


def rng_shuffle(rng, container):
    shuffled_i = list(range(len(container)))
    rng.shuffle(shuffled_i)
    return [container[i] for i in shuffled_i]
