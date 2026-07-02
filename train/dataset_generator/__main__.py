from . import *

if __name__ == "__main__":
    # test_image_overlay()
    test_new_spec()

    l = DataLoader("dataset.priv.yaml")
    print()
    d = DatasetGenerator(
        data_pairs=l.generate_data_pairs(),
    )
    d.debug_dump()
